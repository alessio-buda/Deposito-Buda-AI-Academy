def open_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None

def count_words(file_path):
    pass

def count_lines(file_path):
    pass

def  word_frequency(file_path):
    pass

def main():
    content = open_file("2025-08-18\ex-01\input.txt")
    if content:
        print("File content:")
        print(content)
        
if __name__ == "__main__":
    main()