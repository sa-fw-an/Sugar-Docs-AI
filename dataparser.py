import os
import markdown
from bs4 import BeautifulSoup

def parse_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    html = markdown.markdown(text)
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

def parse_markdown_files(directory):
    data = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                content = parse_markdown_file(file_path)
                relative_path = os.path.relpath(file_path, directory)
                data[relative_path] = content
    return data

def save_content_by_topic(data, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for file, content in data.items():
        topic_dir = os.path.join(output_directory, os.path.dirname(file))
        if not os.path.exists(topic_dir):
            os.makedirs(topic_dir)
        output_file_path = os.path.join(topic_dir, os.path.basename(file).replace('.md', '.txt'))
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(content)

def main():
    src_directories = ['data/sugar-docs/src', 'data/music-blocks']
    output_directory = 'parsed_data'
    for src_directory in src_directories:
        data = parse_markdown_files(src_directory)
        save_content_by_topic(data, output_directory)
    print(f"Parsed content saved in {output_directory}")

if __name__ == "__main__":
    main()