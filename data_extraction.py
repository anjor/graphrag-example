import PyPDF2


def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    with open(pdf_path, "rb") as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Get the number of pages in the PDF
        num_pages = len(pdf_reader.pages)

        # Initialize an empty string to store the extracted text
        text = ""

        # Iterate through all pages and extract text
        for page_num in range(num_pages):
            # Get the page object
            page = pdf_reader.pages[page_num]

            # Extract text from the page
            page_text = page.extract_text()

            # Append the extracted text to the overall text
            text += page_text + "\n\n"  # Add newlines between pages

    return text


# Example usage
pdf_path = "./0704.0044v4.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
print(extracted_text)

# Optionally, save the extracted text to a file
with open("./ragtest/input/extracted_text.txt", "w", encoding="utf-8") as output_file:
    output_file.write(extracted_text)
