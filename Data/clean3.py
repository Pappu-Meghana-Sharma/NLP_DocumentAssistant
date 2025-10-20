import re

def clean_telugu_text(input_filepath, output_filepath):
    
    try:
        with open(input_filepath, "r", encoding="utf-8") as file:
            text = file.read()
        
        print("Starting bulk data cleaning process...")

        text = re.sub(r'---.*?---|==.*?==', '', text, flags=re.DOTALL)
        
        text = re.sub(r'\[\[(.*?)\]\]', r'\1', text) 
        text = re.sub(r'\{\{(.*?)\}\}', '', text)    
        text = re.sub(r'<[^>]*>', '', text)          

        text = re.sub(r'[a-zA-Z]+', '', text, flags=re.UNICODE)
        text = re.sub(r'[\u0900-\u097F]+', '', text, flags=re.UNICODE)  
        text = re.sub(r'[0-9+\-*/=()]+', '', text, flags=re.UNICODE)
        
        cleaned_text = re.sub(r'[^\u0C00-\u0C7F\s\.,]', '', text, flags=re.UNICODE)
        
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        print("Bulk cleaning complete. Starting line-by-line filtering...")
        
        lines = cleaned_text.split('.')
        
        junk_line_pattern = re.compile(r'^\s*([\u0C00-\u0C7F]{1,2}\s*[,\.]?\s*|\s*,\s*\.\s*|\s*\.\s*)$', flags=re.UNICODE)
        
        filtered_lines = []
        for line in lines:
            line = line.strip()
            
            if not line or junk_line_pattern.match(line):
                continue
            
            words = re.findall(r'[\u0C00-\u0C7F]+', line)
            
            if len(words) <= 2:
                continue
            
            line = ' '.join([w for w in words if len(w) > 1])
            
            if line:  
                filtered_lines.append(line)
        
        final_text = '.\n'.join(filtered_lines) + '.'

        final_text = re.sub(r'\.{2,}', '.', final_text)

        with open(output_filepath, "w", encoding="utf-8") as file:
            file.write(final_text)

        print(f"Data cleaning complete. Cleaned data saved to '{output_filepath}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = "merged_telugu_wikipedia_data.txt" 
    output_file = "final_cleaned_telugu_data_2.txt"
    
    clean_telugu_text(input_file, output_file)
