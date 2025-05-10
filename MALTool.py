import os
# import csv # No longer needed
from openai import OpenAI
import re # Import regular expression module for filename sanitization
import time # Import time for potential delays

# --- Konfiguration ---
# Es ist sicherer, den API-Schlüssel als Umgebungsvariable zu setzen,
# aber hier wird er direkt verwendet, wie im Originalskript.
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-2.5-pro-preview-03-25" # Oder ein anderes Modell deiner Wahl, z.B. "openai/gpt-4-turbo" für bessere Ergebnisse bei komplexen Prompts
# INPUT_CSV_FILE = "input_data.csv" # No longer needed
OUTPUT_FOLDER = "output_Toolevaluation" # Name des Ordners für die Ausgabe-Textdateien
# --- Ende Konfiguration ---

# --- Hardcoded Tool List ---
# Liste von Dictionaries, jedes enthält Name, Beschreibung und Link
tool_list = [
    {
        "name": "Sirius",
        "description": "A framework for building domain-specific modeling workbenches. It provides a graphical environment for creating and editing models, and it can be used to create custom editors for various modeling languages.",
        "link": "https://eclipse.dev/sirius/doc/"
    }
]
# --- Ende Hardcoded Tool List ---


# Initialisiere den OpenAI Client für OpenRouter
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def sanitize_filename(name: str) -> str:
    """Bereinigt einen String, um ihn als Dateinamen verwenden zu können."""
    # Entferne oder ersetze spezifische Teile wie "(Sirius-basiert)"
    name = name.replace("(Sirius-basiert)", "").replace("(EMF-Based)", "").strip()
    name = re.sub(r'\(.*?\)', '', name).strip() # Entferne alles in Klammern zur Sicherheit

    # Entferne ungültige Zeichen für Dateinamen
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    # Ersetze Leerzeichen durch Unterstriche
    name = name.replace(" ", "_").replace("/", "_").replace(".", "")
    # Begrenze die Länge, falls nötig (optional)
    max_len = 60 # Kürzere Länge für bessere Lesbarkeit
    if len(name) > max_len:
        name = name[:max_len]
    # Stelle sicher, dass der Name nicht leer ist
    if not name:
        name = "unnamed_file"
    # Entferne abschließende Unterstriche oder Punkte
    name = name.strip('_.')
    return name

def generate_response(prompt: str, model=MODEL, temperature=0.7, max_tokens=1000000): # max_tokens erhöht
    """Sendet den Prompt an die OpenRouter API und gibt die Antwort zurück."""
    print(f"   -> Sende Anfrage an Modell: {model} (max_tokens={max_tokens})")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant following instructions precisely to evaluate tools based on the provided criteria."
                },
                {
                    "role": "user",
                    "content": prompt # Der detaillierte Prompt wird hier übergeben
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        output = response.choices[0].message.content.strip()
        # Kurze Vorschau der Antwort im Log
        print(f"   -> Antwort erhalten (erste 80 Zeichen): {output[:80]}{'...' if len(output) > 80 else ''}")
        # Füge eine kleine Verzögerung hinzu, um Rate Limits zu vermeiden (optional)
        time.sleep(1) # Warte 1 Sekunde
        return output

    except Exception as e:
        # Detailliertere Fehlerausgabe
        print(f"   -> Fehler bei der API-Anfrage für Prompt beginnend mit '{prompt[:50]}...': {e}")
        # Füge eine längere Verzögerung nach einem Fehler hinzu
        time.sleep(5) # Warte 5 Sekunden nach einem Fehler
        return f"Error during API call: {e}" # Gibt Fehler im Output zurück

# Funktion angepasst, um die hardcodierte Liste zu verwenden
def process_tool_list(tools: list, output_dir: str):
    """Verarbeitet die Tool-Liste, generiert Antworten und speichert sie."""

    # Erstelle den Output-Ordner, falls er nicht existiert
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output wird in Ordner '{output_dir}' gespeichert.")
    print(f"Verarbeite insgesamt {len(tools)} Werkzeuge.")

    # Iteriere durch die hardcodierte Liste
    for i, tool in enumerate(tools):
        name = tool.get("name", "N/A").strip()
        link = tool.get("link", "N/A").strip()
        # description = tool.get("description", "N/A").strip() # Beschreibung wird aktuell nicht im Prompt verwendet

        print(f"\n--- Verarbeite Werkzeug {i+1}/{len(tools)} ---")
        print(f"   Name: '{name}'")
        print(f"   Link: '{link}'")

        if name == "N/A" or link == "N/A":
            print("   -> Übersprungen: Name oder Link fehlt in den hardcodierten Daten.")
            continue

        # --- Prompt-Erstellung ---
        # Der detaillierte Prompt wird hier als f-string eingefügt
        # Name und Link werden aus dem aktuellen 'tool' Dictionary geholt
        prompt = f"""🧑‍💻 Role

You are a senior technical analyst with domain expertise in cybersecurity modeling, threat simulation, and evaluation of tooling for security analysis. You have specialized knowledge in the Meta Attack Language (MAL) and hands-on experience with modeling frameworks, simulation engines, and visual analytics tools.

🎯 Task Objective

Evaluate the tool {name} to determine its suitability for supporting and visualizing the full workflow of the Meta Attack Language (MAL). Assess the tool’s alignment with key functional, usability, compatibility, and performance requirements defined below. You should not evaluate if there are any MAL specific methods available but how easy or what kind of possiblilties the tools provide to integrate MAL.

📘 Context

Your evaluation should consider the full MAL modeling and simulation lifecycle, including:



Creation and editing of .mal language specifications

Definition of system models (e.g. YAML, JSON)

Attacker profiles

Compilation and simulation of attack graphs

Visualization and interpretation of attack paths

Key Concepts of MAL to Consider:

Assets, attack steps, associations, defenses

Graph-based attack path generation and visualization

Simulation-based risk and security posture analysis

Compatibility with the malc compiler and ecosystem tools

For a refresher, refer to the summarized MAL description provided in the context section of this task.

📥 Inputs for Analysis

Tool Name: {name} 

Tool Website: {link} 

✅ Evaluation Requirements & Grading Criteria

Evaluate the tool against each of the following requirements (HLR-1 to HLR-12). Use the grading scale and provide detailed, evidence-backed justifications.

Example Format (for each requirement):



Requirement: HLR-1 — Functional Suitability: Support for Full MAL Workflow

Evaluation Score: 0 / 0.5 / 1

Justification: [Detailed reasoning, referring to tool features, documentation, limitations, etc.]

Req IDCategoryRequirement DescriptionScoring CriteriaHLR-1FunctionalTool should support creation/editing of .mal files, model instances, and simulation initiation/viewing0 / 0.5 / 1HLR-2FunctionalIntegration with MAL tools (e.g. compiler/simulator)0 / 0.5 / 1HLR-3VisualizationGraph-based features: zoom, abstraction, grouping, visual customization0 / 0.5 / 1HLR-4Pattern ReuseAbility to define, reuse graph motifs and attack patterns0 / 0.5 / 1HLR-5DocumentationBuilt-in guides, tutorials, and structured workflows0 / 0.5 / 1HLR-6UsabilityIntuitive GUI, drag-drop, dual (textual/graphical) views, syntax validation0 / 0.5 / 1HLR-7CollaborationReal-time multi-user collaboration and Git/VCS integration0 / 0.5 / 1HLR-8CompatibilitySupport for import/export in MAL-compatible formats0 / 0.5 / 1HLR-9MaintainabilityOpen-source, community-driven, sustainable development model0 / 1HLR-10PerformanceHandles large models and complex graphs without UI or system degradation0 / 0.5 / 1HLR-11PortabilityCross-platform or web-based ease of use, low barrier to installation0 / 0.5 / 1HLR-12SecurityOptions for local/on-premise data storage for secure environments0 / 1



🔍 Instructions for Evaluation

Information Gathering

Review the tool’s official website and documentation.

Conduct supplementary web research for user reviews, academic papers, tutorials, and community feedback.

Search for any references to usage of the tool with MAL, DSLs, graph modeling, or attack simulation.

Evaluation

Assign a score (0, 0.5, or 1) to each requirement based on findings.

If insufficient data is available for a requirement, clearly state so.

Justification

Provide detailed reasoning for each score using facts from your research.

Include specific feature references, screenshots (if relevant), or quotes from documentation to support your assessment.

🧾 Desired Output Format

For each requirement:



less

CopyEdit

### Requirement: HLR-1 – [Title]

**Evaluation**: [0 | 0.5 | 1]

**Justification**: [Clear explanation with reference to findings or limitations]



...[Repeat for all HLRs]

🔚 Optional Final Summary

At the end of the evaluation, provide a concise summary:



Overall suitability for use with MAL

Notable strengths and weaknesses

Whether the tool functions best as a standalone solution or an enabler within a larger toolchain
"""
        # --- Ende Prompt-Erstellung ---

        # Generiere die Antwort von der KI
        ai_response = generate_response(prompt) # max_tokens wird in der Funktion gesetzt

        # Erstelle einen sicheren Dateinamen aus dem 'Name'
        output_filename = sanitize_filename(name) + ".txt"
        output_path = os.path.join(output_dir, output_filename)

        # Speichere die Antwort in einer Textdatei
        try:
            with open(output_path, "w", encoding="utf-8") as outfile:
                outfile.write(ai_response)
            print(f"   -> Antwort gespeichert in: '{output_path}'")
        except IOError as e:
            print(f"   -> Fehler beim Schreiben der Datei '{output_path}': {e}")
        except Exception as e:
             print(f"   -> Unerwarteter Fehler beim Schreiben der Datei '{output_path}': {e}")
                # --- HIER DIE VERZÖGERUNG EINFÜGEN ---
        

    print(f"\nVerarbeitung von {len(tools)} Werkzeugen abgeschlossen.")


if __name__ == "__main__":
    print("Starte Skript...")
    # Rufe die Funktion auf, die die hardcodierte Liste verarbeitet
    process_tool_list(tool_list, OUTPUT_FOLDER)
    print("\nSkript beendet.")