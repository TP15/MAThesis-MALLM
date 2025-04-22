import MalCompiler

def compile(lang_file: str, output_file: str) -> None:
    """Compile language and dump into output file"""
    compiler = MalCompiler()
    with open(output_file, "w") as f:
        json.dump(compiler.compile(lang_file), f, indent=2)