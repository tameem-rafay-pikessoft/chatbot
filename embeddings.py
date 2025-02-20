import os
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google Drive API 
from googleapiclient.discovery import build
from google.oauth2 import service_account
from io import BytesIO

# Document processing
import PyPDF2
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# Configuration
SCOPES = ['https://www.googleapis.com/auth/drive']  # Full access to Drive
FOLDER_ID = "1tE8CYDBCQCEeU-Kar5iQs2Pn7qfEz_S1"  # Add your Google Drive folder ID here after sharing with the service account
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
# COLLECTION_NAME = "embeddings"  # Qdrant collection name
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2 model
PERSIST_DIRECTORY = "chroma_db"  # Local directory to store Chroma DB

class DocumentProcessor:
    def __init__(self, folder_id: str, 
                #  qdrant_url: str, qdrant_api_key: str,
                persist_directory: str = PERSIST_DIRECTORY,
                 chunk_size: int = CHUNK_SIZE, 
                 chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the document processor.
        
        Args:
            folder_id: Google Drive folder ID to monitor
            persist_directory: Directory to store Chroma DB
            chunk_size: Size of text chunks for embeddings
            chunk_overlap: Overlap between chunks
        """
        self.folder_id = folder_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Chroma vector store
        self.vector_store = self._init_chroma()
    
        # Initialize Google Drive service
        self.drive_service = self._authenticate_drive()
    
        # Track document metadata
        self.document_metadata = self._load_document_metadata()
    
   
    def _authenticate_drive(self):
        """Authenticate with Google Drive API using service account."""
        # Path to your service account credentials file
        credentials_path = 'credentials.json'
        
        # Create credentials object from service account file
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=SCOPES
        )
        
        # Build and return the Drive service
        return build('drive', 'v3', credentials=credentials)
    
    def _load_document_metadata(self) -> Dict:
        """Load document metadata from storage."""
        metadata_path = os.path.join("metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}  # Return empty dict if no metadata file exists
    
    def _save_document_metadata(self):
        """Save document metadata to storage."""
        metadata_path = os.path.join("metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(self.document_metadata, f)
    
    def _get_drive_files(self) -> List[Dict]:
        """Get all files from the specified Google Drive folder."""
        try:
            print(f"Searching for files in folder: {self.folder_id}")
            
            # First try to get the folder details to verify access
            folder = self.drive_service.files().get(fileId=self.folder_id).execute()
            print(f"Successfully accessed folder: {folder.get('name', 'Unknown')}")
            
            # Now get the files
            results = self.drive_service.files().list(
                q=f"'{self.folder_id}' in parents and trashed=false",
                fields="files(id, name, mimeType, modifiedTime)",
                pageSize=100  # Increase page size to make sure we get all files
            ).execute()
            
            files = results.get('files', [])
            print(f"Found {len(files)} files in the folder")
            return files
            
        except Exception as e:
            print(f"Error accessing Google Drive: {str(e)}")
            return []
    
    def _download_file(self, file_id: str) -> BytesIO:
        """Download a file from Google Drive."""
        request = self.drive_service.files().get_media(fileId=file_id)
        file_content = BytesIO()
        downloader = request.execute()
        file_content.write(downloader)
        file_content.seek(0)
        return file_content
    
    def _extract_text(self, file_content: BytesIO, mime_type: str) -> str:
        """Extract text from various file types."""
        if mime_type == 'application/pdf':
            reader = PyPDF2.PdfReader(file_content)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            doc = docx.Document(file_content)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        
        elif mime_type == 'text/plain':
            return file_content.read().decode('utf-8')
        
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")
    
    def _chunk_text(self, text: str, document_id: str, document_name: str) -> List[Dict]:
        """Split text into chunks with metadata."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = text_splitter.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "metadata": {
                    "document_id": document_id,
                    "document_name": document_name,
                    "chunk_id": i,
                    "processed_at": datetime.now().isoformat()
                }
            })
        
        return documents
    
    def _init_chroma(self) -> Chroma:
        """Initialize Chroma vector store."""
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize Chroma
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def _delete_document_embeddings(self, document_id: str):
        """Delete all embeddings for a specific document."""
        try:
            # Get all embeddings for the document
            results = self.vector_store._collection.get(
                where={"document_id": document_id}
            )
            
            if results and results['ids']:
                # Delete all embeddings associated with the document
                self.vector_store._collection.delete(
                    ids=results['ids']
                )
                print(f"Successfully deleted {len(results['ids'])} embeddings for document {document_id}")
            else:
                print(f"No embeddings found for document {document_id}")
                
            # Persist changes to disk
            self.vector_store.persist()
            
        except Exception as e:
            print(f"Error deleting embeddings for document {document_id}: {str(e)}")

    def process_folder(self):
        """
        Process all documents in the folder, creating or updating embeddings as needed.
        """
        # Get all files in the folder
        files = self._get_drive_files()
        updated = False
        
        for file in files:
            #print the name of file 
            print(file)
            file_id = file['id']
            file_name = file['name']
            mime_type = file['mimeType']
            modified_time = file['modifiedTime']
            
            # Check if we need to process this file
            if file_id in self.document_metadata:
                if self.document_metadata[file_id]['modified_time'] == modified_time:
                    # File hasn't changed, skip processing
                    print(f"No changes detected for {file_name}")
                    continue
                else:
                    # File has been updated, delete old embeddings
                    print(f"Document updated: {file_name}")
                    self._delete_document_embeddings(file_id)
            else:
                print(f"New document found: {file_name}")
            
            try:
                # Supported file types
                if mime_type in ['application/pdf', 
                                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                                'text/plain']:
                    
                    # Download and extract text
                    file_content = self._download_file(file_id)
                    text = self._extract_text(file_content, mime_type)
                    
                    # Chunk text
                    chunks = self._chunk_text(text, file_id, file_name)
                    print(text)
                    
                    # Add to vector store
                    texts = [chunk["text"] for chunk in chunks]
                    metadatas = [{"document_id": file_id, "chunk_id": i, "document_name": file_name} for i, _ in enumerate(texts)]

                    self.vector_store.add_texts(texts=texts, metadatas=metadatas)
                    self.vector_store.persist()  # Make sure to persist after adding



                    # Update metadata
                    self.document_metadata[file_id] = {
                        'name': file_name,
                        'modified_time': modified_time,
                        'processed_at': datetime.now().isoformat(),
                        'chunk_count': len(chunks)
                    }
                    
                    updated = True
                    print(f"Processed {file_name}: {len(chunks)} chunks created")
                else:
                    print(f"Skipping unsupported file type: {file_name} ({mime_type})")
            
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
        
        # Save metadata if any updates were made
        if updated:
            self._save_document_metadata()
            print("Metadata updated.")
        else:
            print("No changes detected in any documents.")
        
        return updated

 

def main():
    """Main execution function."""

    processor = DocumentProcessor(
        folder_id=FOLDER_ID,
        persist_directory=PERSIST_DIRECTORY,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Process documents
    processor.process_folder()



if __name__ == "__main__":
    main()