from facenet_pytorch import MTCNN
import sys

# Capture the standard output to a variable
original_stdout = sys.stdout
output_text = None

try:
    with open('mtcnnhelp.txt', 'w+') as f:
        sys.stdout = f  # Redirect standard output to the file

        # Call the help() function
        help(MTCNN)  # Replace 'print' with the function/module you want to get help for
        
        # Read the captured text from the file
        f.seek(0)
        output_text = f.read()
finally:
    sys.stdout = original_stdout  # Reset standard output to the original value