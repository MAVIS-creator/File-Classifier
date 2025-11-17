"""
GUI File Classifier
Interactive graphical interface for file classification
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont
import os
from PIL import Image, ImageTk
from file_classifier import train_classifier, evaluate_classifier, generate_accuracy_report
from data_handler import prepare_data
from hash_features import extract_hash_features
import pandas as pd


class FileClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🛡️ File Type Classifier - MD5 Hash Features")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        self.classifier = None
        self.selected_files = []
        
        # Custom fonts
        self.title_font = tkfont.Font(family="Arial", size=16, weight="bold")
        self.header_font = tkfont.Font(family="Arial", size=12, weight="bold")
        self.normal_font = tkfont.Font(family="Arial", size=10)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50", pady=15)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            title_frame,
            text="🛡️ File Type Classifier Using Hash Features",
            font=self.title_font,
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Naive Bayes Classifier | MD5 Hash Metadata Analysis",
            font=("Arial", 9),
            bg="#2c3e50",
            fg="#ecf0f1"
        )
        subtitle_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#f0f0f0", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Step 1: Train Model
        train_frame = tk.LabelFrame(
            main_frame,
            text="Step 1: Train Model",
            font=self.header_font,
            bg="#ecf0f1",
            padx=15,
            pady=15
        )
        train_frame.pack(fill=tk.X, pady=(0, 15))
        
        train_btn = tk.Button(
            train_frame,
            text="🚀 Train Classifier on Sample Data",
            command=self.train_model,
            font=self.normal_font,
            bg="#27ae60",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        train_btn.pack()
        
        self.train_status = tk.Label(
            train_frame,
            text="Status: Not trained",
            font=self.normal_font,
            bg="#ecf0f1",
            fg="#7f8c8d"
        )
        self.train_status.pack(pady=(10, 0))
        
        # Step 2: Select Files
        file_frame = tk.LabelFrame(
            main_frame,
            text="Step 2: Select Files to Classify",
            font=self.header_font,
            bg="#ecf0f1",
            padx=15,
            pady=15
        )
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        select_btn = tk.Button(
            file_frame,
            text="📁 Select Files...",
            command=self.select_files,
            font=self.normal_font,
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        select_btn.pack()
        
        self.file_listbox = tk.Listbox(
            file_frame,
            font=("Courier New", 9),
            height=6,
            bg="white",
            selectmode=tk.SINGLE
        )
        self.file_listbox.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Step 3: Classify
        classify_frame = tk.LabelFrame(
            main_frame,
            text="Step 3: Run Classification",
            font=self.header_font,
            bg="#ecf0f1",
            padx=15,
            pady=15
        )
        classify_frame.pack(fill=tk.X, pady=(0, 15))
        
        classify_btn = tk.Button(
            classify_frame,
            text="🔍 Classify Selected Files",
            command=self.classify_files,
            font=self.normal_font,
            bg="#e67e22",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        classify_btn.pack()
        
        # Results
        results_frame = tk.LabelFrame(
            main_frame,
            text="Classification Results",
            font=self.header_font,
            bg="#ecf0f1",
            padx=15,
            pady=15
        )
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Results text area with scrollbar
        scroll = tk.Scrollbar(results_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(
            results_frame,
            font=("Courier New", 9),
            height=8,
            bg="white",
            yscrollcommand=scroll.set
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        scroll.config(command=self.results_text.yview)
        
        # View Report Button
        view_btn = tk.Button(
            main_frame,
            text="📊 View Accuracy Report & Visualization",
            command=self.view_report,
            font=self.normal_font,
            bg="#9b59b6",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        view_btn.pack(pady=(0, 10))
        
    def train_model(self):
        try:
            self.train_status.config(text="Status: Training...", fg="#e67e22")
            self.root.update()
            
            # Prepare data
            X_train, X_test, y_train, y_test = prepare_data()
            
            # Train
            self.classifier = train_classifier(X_train, y_train)
            
            # Evaluate
            from file_classifier import evaluate_classifier, visualize_results
            metrics = evaluate_classifier(self.classifier, X_test, y_test)
            
            # Generate visualization
            visualize_results(metrics)
            
            # Update status
            accuracy = metrics['accuracy'] * 100
            self.train_status.config(
                text=f"Status: ✓ Trained! Accuracy: {accuracy:.2f}%",
                fg="#27ae60"
            )
            
            messagebox.showinfo(
                "Training Complete",
                f"Model trained successfully!\n\n"
                f"Accuracy: {accuracy:.2f}%\n"
                f"Precision: {metrics['precision']:.4f}\n"
                f"Recall: {metrics['recall']:.4f}\n"
                f"F1-Score: {metrics['f1_score']:.4f}\n\n"
                f"Visualization saved as 'classification_results.png'"
            )
            
        except Exception as e:
            self.train_status.config(text="Status: Error during training", fg="#e74c3c")
            messagebox.showerror("Training Error", f"Failed to train model:\n{str(e)}")
    
    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select Files to Classify",
            filetypes=[
                ("All Files", "*.*"),
                ("Executables", "*.exe"),
                ("DLL Files", "*.dll"),
                ("Text Files", "*.txt"),
                ("Python Files", "*.py"),
                ("Images", "*.png;*.jpg;*.jpeg"),
                ("Archives", "*.zip;*.rar")
            ]
        )
        
        if files:
            self.selected_files = list(files)
            self.file_listbox.delete(0, tk.END)
            for f in self.selected_files:
                self.file_listbox.insert(tk.END, os.path.basename(f))
    
    def classify_files(self):
        if not self.classifier:
            messagebox.showwarning(
                "Not Trained",
                "Please train the model first (Step 1)!"
            )
            return
        
        if not self.selected_files:
            messagebox.showwarning(
                "No Files",
                "Please select files to classify (Step 2)!"
            )
            return
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Classification Results:\n")
        self.results_text.insert(tk.END, "=" * 70 + "\n\n")
        
        benign_count = 0
        malicious_count = 0
        
        for file_path in self.selected_files:
            try:
                features = extract_hash_features(file_path)
                if not features:
                    self.results_text.insert(
                        tk.END,
                        f"❌ {os.path.basename(file_path)}: Error extracting features\n\n"
                    )
                    continue
                
                df = pd.DataFrame([features]).fillna(0)
                prediction = self.classifier.predict(df)[0]
                
                if prediction == 1:
                    result = "🔴 MALICIOUS"
                    malicious_count += 1
                    color_tag = "malicious"
                else:
                    result = "🟢 BENIGN"
                    benign_count += 1
                    color_tag = "benign"
                
                self.results_text.insert(tk.END, f"File: {os.path.basename(file_path)}\n")
                self.results_text.insert(tk.END, f"Path: {file_path}\n")
                self.results_text.insert(tk.END, f"Prediction: {result}\n", color_tag)
                self.results_text.insert(tk.END, "-" * 70 + "\n\n")
                
            except Exception as e:
                self.results_text.insert(
                    tk.END,
                    f"❌ {os.path.basename(file_path)}: Error - {str(e)}\n\n"
                )
        
        # Summary
        self.results_text.insert(tk.END, "\nSummary:\n")
        self.results_text.insert(tk.END, f"Total Files: {len(self.selected_files)}\n")
        self.results_text.insert(tk.END, f"Benign: {benign_count}\n", "benign")
        self.results_text.insert(tk.END, f"Malicious: {malicious_count}\n", "malicious")
        
        # Configure tags for colors
        self.results_text.tag_config("benign", foreground="#27ae60", font=("Courier New", 9, "bold"))
        self.results_text.tag_config("malicious", foreground="#e74c3c", font=("Courier New", 9, "bold"))
        
        messagebox.showinfo("Classification Complete", f"Classified {len(self.selected_files)} file(s)")
    
    def view_report(self):
        report_path = "classification_results.png"
        
        if not os.path.exists(report_path):
            messagebox.showwarning(
                "No Report",
                "Please train the model first to generate the report!"
            )
            return
        
        # Create new window
        report_window = tk.Toplevel(self.root)
        report_window.title("📊 Accuracy Report & Visualization")
        report_window.geometry("1000x600")
        report_window.configure(bg="white")
        
        # Title
        title = tk.Label(
            report_window,
            text="Classification Accuracy Report",
            font=self.title_font,
            bg="white"
        )
        title.pack(pady=10)
        
        # Load and display image
        try:
            img = Image.open(report_path)
            # Resize to fit window
            img.thumbnail((950, 500), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            img_label = tk.Label(report_window, image=photo, bg="white")
            img_label.image = photo  # Keep reference
            img_label.pack(pady=10)
            
        except Exception as e:
            error_label = tk.Label(
                report_window,
                text=f"Error loading image: {str(e)}",
                font=self.normal_font,
                bg="white",
                fg="red"
            )
            error_label.pack(pady=20)
        
        # Close button
        close_btn = tk.Button(
            report_window,
            text="Close",
            command=report_window.destroy,
            font=self.normal_font,
            bg="#95a5a6",
            fg="white",
            padx=20,
            pady=5
        )
        close_btn.pack(pady=10)


def main():
    root = tk.Tk()
    app = FileClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
