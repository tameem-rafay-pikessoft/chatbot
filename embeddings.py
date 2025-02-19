import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google Drive API 
from googleapiclient.discovery import build
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from io import BytesIO

# Document processing
import PyPDF2
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http import models

# Configuration 
SCOPES = ['https://www.googleapis.com/auth/drive']  # Full access to Drive
FOLDER_ID = "1tE8CYDBCQCEeU-Kar5iQs2Pn7qfEz_S1"  # Add your Google Drive folder ID here after sharing with the service account
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI embedding model
COLLECTION_NAME = "embeddings"  # Qdrant collection name

class DocumentProcessor:
    def __init__(self, api_key: str, folder_id: str, 
                 qdrant_url: str, qdrant_api_key: str,
                 chunk_size: int = CHUNK_SIZE, 
                 chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the document processor.
        
        Args:
            api_key: OpenAI API key for embeddings
            folder_id: Google Drive folder ID to monitor
            qdrant_url: Qdrant cloud service URL
            qdrant_api_key: Qdrant API key
            chunk_size: Size of text chunks for embeddings
            chunk_overlap: Overlap between chunks
        """
        self.folder_id = folder_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        
        # Initialize embeddings
        os.environ["OPENAI_API_KEY"] = api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
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
    
    def _initialize_vector_store(self):
        """Initialize or load the vector store."""
        # Create Qdrant client for cloud
        client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
        # Create collection if it doesn't exist
        collections = client.get_collections().collections
        collection_exists = any(col.name == COLLECTION_NAME for col in collections)
        
        if not collection_exists:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=1536,  # OpenAI embeddings dimension
                    distance=models.Distance.COSINE
                )
            )
        
        return Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=self.embeddings
        )
    
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
    
    def _delete_document_embeddings(self, document_id: str):
        """Delete all embeddings for a specific document."""
        self.vector_store.delete(filter={"document_id": document_id})
    
    def process_folder(self):
        """
        Process all documents in the folder, creating or updating embeddings as needed.
        """
        # Get all files in the folder
        files = self._get_drive_files()
        print(files)
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
                    
                    # Add to vector store
                    texts = [chunk["text"] for chunk in chunks]
                    metadatas = [chunk["metadata"] for chunk in chunks]
                    
                    self.vector_store.add_texts(texts=texts, metadatas=metadatas)
                    
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
    # Load environment variables
    load_dotenv()
    
    # Debug environment variables
    print("Environment variables:")
    print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print(f"QDRANT_URL: {'Set' if os.getenv('QDRANT_URL') else 'Not set'}")
    print(f"QDRANT_API_KEY: {'Set' if os.getenv('QDRANT_API_KEY') else 'Not set'}")
    
    # Get API key from environment or config
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize processor
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables not set")
    
    processor = DocumentProcessor(
        api_key=api_key,
        folder_id=FOLDER_ID,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Process documents
    processor.process_folder()

if __name__ == "__main__":
    main()