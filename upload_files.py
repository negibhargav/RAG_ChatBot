import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

# Target directory
UPLOAD_DIR = "Uploaded_document"

# Supported file extensions (lowercase for comparison)
ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.doc', '.docx', '.txt'}

# Mapping of file extensions to their corresponding subfolders
EXTENSION_TO_SUBDIR = {
    '.pdf': 'pdfs',
    '.doc': 'docus',
    '.docx': 'docus',
    '.png': 'images',
    '.jpg': 'images',
    '.jpeg': 'images',
    '.txt': 'text'
}

def setup_gui():
    """Hides the main Tkinter window."""
    root = tk.Tk()
    root.withdraw()
    # Bring the dialog to the front
    root.attributes('-topmost', True)
    return root

def upload_documents():
    """Opens a file dialog to upload supported files into the Uploaded_document folder."""
    print("=== Document Uploader Module ===")
    
    # Ensure the upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    print(f"Target upload folder configured: '{UPLOAD_DIR}/'")
    
    root = setup_gui()
    
    # Define file filters for the GUI dialog
    file_filters = [
        ("Allowed Documents/Images", "*.pdf;*.png;*.jpg;*.jpeg;*.doc;*.docx;*.txt"),
        ("PDF Files (*.pdf)", "*.pdf"),
        ("Image Files (*.png, *.jpg, *.jpeg)", "*.png;*.jpg;*.jpeg"),
        ("Word Documents (*.doc, *.docx)", "*.doc;*.docx"),
        ("Text Files (*.txt)", "*.txt"),
        ("All Files (*.*)", "*.*")
    ]
    
    print("\nOpening file selection dialog... Please choose one or more files.")
    
    # Open file dialog (allows multiple selection)
    selected_files = filedialog.askopenfilenames(
        title="Select Files to Upload",
        filetypes=file_filters
    )
    
    if not selected_files:
        print("No files were selected.")
        root.destroy()
        return []
    
    print(f"\nProcessing {len(selected_files)} selected file(s)...")
    
    uploaded_files = []
    skipped_count = 0
    
    for file_path in selected_files:
        filename = os.path.basename(file_path)
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        # Validation
        if ext not in ALLOWED_EXTENSIONS:
            print(f"[-] Skipped: '{filename}' - Unsupported file type ({ext}).")
            skipped_count += 1
            continue
            
        # Determine the subdirectory based on file extension
        subdir = EXTENSION_TO_SUBDIR.get(ext, '')
        target_dir = os.path.join(UPLOAD_DIR, subdir)
        
        # Ensure the type-specific subdirectory exists
        os.makedirs(target_dir, exist_ok=True)
        
        destination_path = os.path.join(target_dir, filename)
        
        try:
            # Copy file
            shutil.copy2(file_path, destination_path)
            print(f"[+] Successfully uploaded: '{filename}' to subfolder '{subdir}/'")
            uploaded_files.append(f"{subdir}/{filename}")
        except Exception as e:
            print(f"[x] Error uploading '{filename}': {e}")
            skipped_count += 1
            
    print("\n=== Upload Summary ===")
    print(f"Total files selected: {len(selected_files)}")
    print(f"Successfully uploaded: {len(uploaded_files)}")
    print(f"Skipped / Failed: {skipped_count}")
    
    # Inform user via dialog box
    messagebox.showinfo(
        "Upload Complete", 
        f"Successfully uploaded {len(uploaded_files)} file(s) to '{UPLOAD_DIR}/'.\nSkipped/failed: {skipped_count}."
    )
    
    # Destroy the root window to clean up
    root.destroy()
    return uploaded_files

if __name__ == "__main__":
    upload_documents()
