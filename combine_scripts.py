import os
from pathlib import Path

def combine_files_to_txt(target_directory):
    # Set up paths
    folder_path = Path(target_directory)
    output_filename = "combined_scripts_output.txt"
    output_file_path = folder_path / output_filename
    
    # Define file extensions to ignore
    ignored_extensions = {'.csv', '.parquet'}
    
    # Open the output file in write mode
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        # rglob('*') searches all files in the directory and subdirectories.
        # If you ONLY want the top-level folder, change `rglob('*')` to `iterdir()`
        for file_path in folder_path.rglob('*'):
            
            # 1. Skip if it's a directory
            if not file_path.is_file():
                continue
                
            # 2. Skip the output file itself so we don't cause an infinite loop
            if file_path.name == output_filename:
                continue
                
            # 3. Skip ignored extensions
            if file_path.suffix.lower() in ignored_extensions:
                continue
            
            # 4. Skip empty files (0 bytes)
            if file_path.stat().st_size == 0:
                continue
                
            try:
                # Read the content of the file
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    
                # 5. Skip if the file only contains whitespace/blank lines
                if not content.strip():
                    continue
                    
                # Get the extension without the dot for the markdown code block (e.g., 'yaml', 'py')
                ext = file_path.suffix.lstrip('.')
                if not ext:
                    ext = 'text' # Default fallback
                    
                # Write to the output file using your exact requested format
                outfile.write(f"**File Name** - `{file_path}`\n")
                outfile.write("**text** -\n\n")
                outfile.write(f"```{ext}\n")
                outfile.write(content)
                
                # Ensure there's a newline before closing the code block
                if not content.endswith('\n'):
                    outfile.write('\n')
                    
                outfile.write("```\n\n---\n\n")
                
            except UnicodeDecodeError:
                # Silently skip binary files (like images, .pyc, etc.) that can't be read as text
                pass
            except Exception as e:
                print(f"Could not read {file_path.name}: {e}")

    print(f"Done! Combined file saved to: {output_file_path}")

if __name__ == "__main__":
    # os.getcwd() gets the directory where the script is being run from.
    # As long as you run this script from inside the "JPM week 6_1st_version" folder, 
    # it will target that folder.
    current_folder = os.getcwd()
    combine_files_to_txt(current_folder)