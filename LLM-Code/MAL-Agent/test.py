import json

def format_ttc_distribution(ttc_data):
    """
    Formats the Time-to-Compromise distribution for MAL syntax.
    Example JSON: {"type": "Exponential", "probability": 0.1} -> MAL: [Exponential(0.1)]
    """
    if not ttc_data:
        return ""
    
    ttc_type = ttc_data.get("type")
    params_str = ""

    if ttc_type == "Exponential": # As per MAL example, uses 'Exponential'
        if "probability" in ttc_data:
            params_str = str(ttc_data["probability"])
        else:
            return f" [{ttc_type}(PARAMETER_MISSING)]" # Fallback if probability is missing
    elif ttc_type == "Deterministic":
        if "value" in ttc_data:
            params_str = str(ttc_data["value"])
        else:
            return f" [{ttc_type}(PARAMETER_MISSING)]"
    elif ttc_type == "Gamma":
        if "shape" in ttc_data and "scale" in ttc_data:
            params_str = f"{ttc_data['shape']}, {ttc_data['scale']}"
        else:
            return f" [{ttc_type}(PARAMETERS_MISSING)]"
    elif ttc_type == "Bernoulli": # As per JSON Schema for probability-based distribution
        if "probability" in ttc_data:
            params_str = str(ttc_data["probability"])
        else:
            return f" [{ttc_type}(PARAMETER_MISSING)]"
    else: # Unknown TTC type or type not specified
        return "" 
    
    if params_str: # If parameters were successfully formatted
        return f" [{ttc_type}({params_str})]"
    else:
        # This case might occur if a known type was specified but its required parameters were not.
        # The specific PARAMETER_MISSING messages above should catch most of these.
        return ""


def generate_mal_from_json(json_data_str):
    """
    Parses a JSON string representing a MAL model and converts it to .mal plain text.
    """
    try:
        json_data = json.loads(json_data_str)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input. {e}"

    mal_output = []

    # Meta section
    meta = json_data.get("meta", {})
    model_name = meta.get("model_name", "unnamed_model")
    language_version = meta.get("language_version", "unknown_version")
    mal_output.append(f'#id: "{model_name}"')
    mal_output.append(f'#version: "{language_version}"')
    
    # Add a blank line after meta if there's more content (assets or associations)
    if json_data.get("asset_definitions") or json_data.get("association_definitions"):
        mal_output.append("")

    # Category and Asset Definitions
    asset_definitions = json_data.get("asset_definitions", [])
    if asset_definitions:
        # Using "System" as the category name, based on the provided MAL example.
        category_name = "System" 
        mal_output.append(f"category {category_name} {{")

        for asset_def in asset_definitions:
            asset_name = asset_def.get("name", "UnnamedAsset")
            is_abstract = asset_def.get("is_abstract", False)
            extends_asset = asset_def.get("extends") # This will be None if not present

            asset_declaration_parts = ["  "] # Indentation for asset line
            if is_abstract:
                asset_declaration_parts.append("abstract ")
            asset_declaration_parts.append(f"asset {asset_name}")
            if extends_asset:
                asset_declaration_parts.append(f" extends {extends_asset}")
            asset_declaration_parts.append(" {")
            mal_output.append("".join(asset_declaration_parts))

            # Attack Steps
            for attack_step in asset_def.get("attack_steps", []):
                step_name = attack_step.get("name", "unnamed_step")
                step_type_json = attack_step.get("type", "OR") # Default to OR if not specified
                step_type_mal = "|" if step_type_json == "OR" else "&"
                
                ttc_str = format_ttc_distribution(attack_step.get("ttc_distribution"))
                
                mal_output.append(f"    {step_type_mal} {step_name}{ttc_str}")

                for reach in attack_step.get("reaches", []):
                    target_role = reach.get("target_asset_role") # Optional
                    target_step = reach.get("target_attack_step_name", "undefined_target_step")
                    
                    reach_str_parts = ["      -> "] # Indentation for reach line
                    if target_role:
                        reach_str_parts.append(f"{target_role}.")
                    reach_str_parts.append(target_step)
                    mal_output.append("".join(reach_str_parts))

            # Defenses (interpreted as generated OR attack steps indicating bypass/failure)
            for defense in asset_def.get("defenses", []):
                defense_name = defense.get("name", "unnamed_defense")
                for i, reach_if_false in enumerate(defense.get("reaches_if_false", [])):
                    target_role_for_name = reach_if_false.get("target_asset_role", "")
                    target_step_for_name = reach_if_false.get("target_attack_step_name", "unknown_effect")
                    
                    # Sanitize parts of the generated name for MAL compatibility
                    safe_defense_name = defense_name.replace(" ", "_").replace(".", "_")
                    safe_target_role = target_role_for_name.replace(" ", "_").replace(".", "_")
                    safe_target_step = target_step_for_name.replace(" ", "_").replace(".", "_")

                    generated_step_name_parts = [f"bypass_{safe_defense_name}_leads_to_"]
                    if safe_target_role:
                        generated_step_name_parts.append(f"{safe_target_role}_")
                    generated_step_name_parts.append(safe_target_step)
                    # To make it somewhat unique if multiple reaches for same defense
                    if i > 0 :
                         generated_step_name_parts.append(f"_{i}")
                    generated_step_name = "".join(generated_step_name_parts)
                    
                    # Defense failure leading to an opportunity is typically an OR step
                    mal_output.append(f"    | {generated_step_name}") 

                    target_role = reach_if_false.get("target_asset_role")
                    target_step = reach_if_false.get("target_attack_step_name")
                    reach_str_parts = ["      -> "]
                    if target_role:
                        reach_str_parts.append(f"{target_role}.")
                    reach_str_parts.append(target_step)
                    mal_output.append("".join(reach_str_parts))
            
            mal_output.append("  }") # Close asset definition

        mal_output.append("}") # Close category block
        
        # Add a blank line after category if associations follow, matching example style
        if json_data.get("association_definitions"):
            mal_output.append("")

    # Association Definitions
    associations = json_data.get("association_definitions", [])
    if associations:
        mal_output.append("associations {")
        for assoc_def in associations:
            # Use "---" for association name if not provided, as per common MAL practice
            name = assoc_def.get("name")
            if name is None : # Handles missing or explicitly null name
                 name = "---"
            
            end1_asset = assoc_def.get("end1_asset_type", "UnknownAsset")
            end1_role = assoc_def.get("end1_role_name", "unknown_role1")
            end1_mult = assoc_def.get("end1_multiplicity", "*")
            
            end2_asset = assoc_def.get("end2_asset_type", "UnknownAsset")
            end2_role = assoc_def.get("end2_role_name", "unknown_role2")
            end2_mult = assoc_def.get("end2_multiplicity", "*")
            
            mal_output.append(f"  {end1_asset} [{end1_role}] {end1_mult} <-- {name} --> {end2_mult} [{end2_role}] {end2_asset}")
        mal_output.append("}")

    return "\n".join(mal_output)

if __name__ == '__main__':
    # Example Usage:
    # You would replace this with your actual JSON string or load from a file
    example_json_str = """
    {
      "meta": {
        "language_version": "1.0.0",
        "model_name": "org.mal-lang.examplelang.fromjson",
        "description": "A sample model generated from JSON."
      },
      "asset_definitions": [
        {
          "name": "Network",
          "is_abstract": false,
          "attack_steps": [
            {
              "name": "access",
              "type": "OR",
              "reaches": [
                {"target_asset_role": "host", "target_attack_step_name": "connect"}
              ]
            }
          ]
        },
        {
          "name": "Host",
          "is_abstract": false,
          "attack_steps": [
            {"name": "connect", "type": "OR", "reaches": [{"target_attack_step_name": "access"}]},
            {"name": "authenticate", "type": "OR", "reaches": [{"target_attack_step_name": "access"}]},
            {"name": "guessPasswordAttempt", "type": "OR", "reaches": [{"target_attack_step_name": "guessedPasswordState"}]},
            {
              "name": "useGuessedPassword", 
              "type": "OR", 
              "ttc_distribution": {"type": "Exponential", "probability": 0.02},
              "reaches": [{"target_attack_step_name": "authenticate"}]
            },
            {"name": "access", "type": "AND", "reaches": []}
          ],
          "defenses": [
            {
              "name": "Firewall",
              "reaches_if_false": [
                {"target_attack_step_name": "unrestrictedNetworkAccess"}
              ]
            }
          ]
        },
        {
          "name": "User",
          "is_abstract": false,
          "attack_steps": [
            {"name": "attemptPhishing", "type": "OR", "reaches": [{"target_attack_step_name": "phish"}]},
            {
              "name": "phish", 
              "type": "OR", 
              "ttc_distribution": {"type": "Exponential", "probability": 0.1},
              "reaches": [{"target_asset_role": "passwords", "target_attack_step_name": "obtain"}]
            }
          ]
        },
        {
          "name": "Password",
          "is_abstract": false,
          "attack_steps": [
            {
              "name": "obtain", 
              "type": "OR", 
              "reaches": [{"target_asset_role": "host", "target_attack_step_name": "authenticate"}]
            }
          ]
        },
        {
          "name": "AbstractDevice",
          "is_abstract": true,
          "attack_steps": [
            {"name": "genericCompromise", "type": "OR", "reaches": []}
          ]
        },
        {
          "name": "SpecialHost",
          "is_abstract": false,
          "extends": "Host",
          "attack_steps": [
            {"name": "exploitSpecialFeature", "type": "OR", "reaches": [{"target_attack_step_name": "access"}]}
          ]
        }
      ],
      "association_definitions": [
        {
          "name": "NetworkAccess",
          "end1_asset_type": "Network", "end1_role_name": "networks", "end1_multiplicity": "*",
          "end2_asset_type": "Host", "end2_role_name": "hosts", "end2_multiplicity": "*"
        },
        {
          "name": "Credentials",
          "end1_asset_type": "Host", "end1_role_name": "host", "end1_multiplicity": "1",
          "end2_asset_type": "Password", "end2_role_name": "passwords", "end2_multiplicity": "*"
        },
        {
          "end1_asset_type": "User", "end1_role_name": "user", "end1_multiplicity": "1",
          "end2_asset_type": "Password", "end2_role_name": "passwords", "end2_multiplicity": "*"
        }
      ]
    }
    """
    
    generated_mal_code = generate_mal_from_json(example_json_str)
    print(generated_mal_code)

    # To test with your specific JSON, you can load it from the file path where it's stored
    # For example:
    # with open('path_to_your_model.json', 'r') as f:
    #     user_json_str = f.read()
    # mal_output = generate_mal_from_json(user_json_str)
    # print(mal_output)