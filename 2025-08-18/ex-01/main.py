import string

def open_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None

def count_words(file):
    # count number of words in a file, exclude punctuation
    content = file
    content = content.translate(str.maketrans("", "", string.punctuation))
    words = content.split()
    return len(words)

def count_lines(file):
    # count number of lines in a file
    return len(file.splitlines())

def  word_frequency(file_path):
    pass

def main():
    content = open_file("2025-08-18\ex-01\input.txt")
    if content:
        print("File content:")
        print(content)
        
        word_count = count_words(content)
        print(f"Number of words: {word_count}")
        
        line_count = count_lines(content)
        print(f"Number of lines: {line_count}")

if __name__ == "__main__":
    main()