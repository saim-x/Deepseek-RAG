Hey there! 👋 So you want to use this *AI PDF Assistant*? Awesome choice! Let’s break it down step by step—no jargon, I promise!  

---

## 🚀 **What Does It Do?**  
I guess we all know

---

## 🛠️ **Setup Guide**  

### **Step 1: Install Dependencies**  
You’ll need Python (3.8+) and a few libraries. Open your terminal and run:  
```bash  
pip install streamlit langchain-chroma langchain-ollama nomic python-dotx pdfplumber  
```  

### **Step 2: Set Up Ollama**  
This app uses AI models via Ollama. Here’s how to get started:  
1. Download Ollama from [ollama.ai](https://ollama.ai/) and install it.  
2. Run these commands to pull the models (this might take a few minutes):  
```bash  
ollama pull deepseek-r1:1.5b  
ollama pull nomic-embed-text  
```  

### **Step 3: Run the App**  
1. Save the Python script (let’s call it `pdf_assistant.py`).  
2. In your terminal, navigate to the folder where the script is saved.  
3. Run:  
```bash  
streamlit run pdf_assistant.py  
```  
Your browser will open automatically with the app!  

---

## 🖥️ **How to Use It**  

### **Starting Fresh**  
- When you first run the app, you’ll see an empty chat screen.  
- Use the **sidebar on the left** to upload PDFs and tweak settings.  

### **Uploading Documents**  
1. Click "Upload Research PDFs" in the sidebar.  
2. Select one or more PDFs (max 50MB each).  
3. Wait for the progress bar—the app is:  
   - Splitting your PDF into "chunks" (small text pieces)  
   - Storing them in a searchable database  
   - Counting pages and tracking processing time  

---

## 🔍 **Behind the Scenes**  

### **How It Works**  
1. **Processing PDFs**:  
   - Splits documents into 1500-character chunks with 300-character overlaps (so no info gets cut off mid-sentence).  
   - Uses both `PyPDFLoader` and `PDFPlumber` as backup—if one fails, the other tries.  
2. **AI Magic**:  
   - Embeds text using **Nomic AI** for better context understanding.  
   - Searches using "max marginal relevance" (balances keyword matches and context diversity).  

### **Security Stuff**  
- Files are hashed (MD5) to avoid duplicates.  
- Local storage only—your PDFs stay on your machine.  
- Size limits prevent mega-file uploads.  

---

## 📊 **Analytics Dashboard**  
Click "System Analytics" to see:  
- Total processed pages and text chunks  
- Average processing time per document  
- Detailed stats for each PDF (pages, chunks, processing time)  

---

## 🚨 **Troubleshooting**  

**Problem**: "Model not found"  
- Fix: Double-check you ran `ollama pull` for the model name in the error.  

**Problem**: App feels slow  
- Try: Reduce PDF file size or split large documents.  

**Problem**: Answers seem off  
- Try: Lower the "Creativity" slider or rephrase your question.  

---

## 🙌 **Final Notes**  
That's all.
Made something cool with this? Found a bug? Let me know! :D

--- 

> **Note**: Keep your Ollama server running while using the app. If you close it, the AI won’t respond!
