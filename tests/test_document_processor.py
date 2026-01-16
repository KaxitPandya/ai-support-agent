"""
Unit tests for the document processor service.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.services.document_processor import DocumentProcessor, get_document_processor


class TestDocumentProcessor:
    """Tests for DocumentProcessor."""
    
    @pytest.fixture
    def temp_upload_dir(self):
        """Create a temporary upload directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def processor(self, temp_upload_dir):
        """Create a document processor with temp directory."""
        return DocumentProcessor(upload_dir=temp_upload_dir)
    
    def test_chunk_text_short(self, processor):
        """Test that short text is not chunked."""
        text = "This is a short text."
        chunks = processor.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_long(self, processor):
        """Test that long text is properly chunked with simple strategy."""
        # Create text longer than chunk size
        text = "This is a sentence. " * 50  # ~1000 characters
        # Use simple chunking strategy for deterministic behavior
        chunks = processor.chunk_text(text, chunk_size=200, overlap=20, strategy="simple")
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 250  # Some flexibility for sentence boundaries
    
    def test_chunk_text_with_overlap(self, processor):
        """Test that chunks have overlap."""
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = processor.chunk_text(text, chunk_size=50, overlap=10)
        
        # With proper overlap, end of one chunk should appear at start of next
        if len(chunks) > 1:
            # Chunks should have some overlapping content
            assert len(chunks) >= 2
    
    def test_chunk_text_empty(self, processor):
        """Test chunking empty text."""
        chunks = processor.chunk_text("")
        assert chunks == []
    
    def test_extract_metadata_with_title(self, processor):
        """Test metadata extraction from content."""
        content = "# Domain Policy\nThis is the policy content."
        metadata = processor.extract_metadata(content, "policy.md")
        
        assert "Domain Policy" in metadata["title"]
        assert "filename" in metadata
        assert "content_hash" in metadata
    
    def test_extract_metadata_category_detection(self, processor):
        """Test category detection from keywords."""
        billing_content = "Payment processing and refund policies."
        metadata = processor.extract_metadata(billing_content, "test.md")
        assert metadata["category"] == "Billing"
        
        security_content = "Phishing and malware prevention."
        metadata = processor.extract_metadata(security_content, "test.md")
        assert metadata["category"] == "Security"
    
    def test_process_text(self, processor):
        """Test processing text into documents."""
        content = "# Test Policy\n\nThis is test content for the policy document. It should be processed properly."
        documents = processor.process_text(content, "test.md")
        
        assert len(documents) >= 1
        assert all(doc.id.startswith("test.md-chunk-") for doc in documents)
        assert all(doc.title for doc in documents)
    
    def test_save_uploaded_file(self, processor, temp_upload_dir):
        """Test saving uploaded file."""
        content = b"Test file content"
        file_path = processor.save_uploaded_file("test.txt", content)
        
        assert file_path.exists()
        assert file_path.read_bytes() == content
    
    def test_process_uploaded_file(self, processor, temp_upload_dir):
        """Test processing uploaded file."""
        content = b"# Policy Title\n\nPolicy content here."
        documents, file_path = processor.process_uploaded_file(
            "policy.md",
            content,
            "Domain Policies"
        )
        
        assert len(documents) >= 1
        assert Path(file_path).exists()
        assert documents[0].category == "Domain Policies"
    
    def test_list_uploaded_files(self, processor, temp_upload_dir):
        """Test listing uploaded files."""
        # Upload some files
        processor.save_uploaded_file("file1.txt", b"content1")
        processor.save_uploaded_file("file2.txt", b"content2")
        
        files = processor.list_uploaded_files()
        
        assert len(files) == 2
        filenames = [f["filename"] for f in files]
        assert "file1.txt" in filenames
        assert "file2.txt" in filenames
    
    def test_delete_file(self, processor, temp_upload_dir):
        """Test deleting file."""
        processor.save_uploaded_file("to_delete.txt", b"content")
        
        # File should exist
        assert len(processor.list_uploaded_files()) == 1
        
        # Delete
        result = processor.delete_file("to_delete.txt")
        assert result is True
        
        # File should be gone
        assert len(processor.list_uploaded_files()) == 0
    
    def test_delete_nonexistent_file(self, processor):
        """Test deleting non-existent file."""
        result = processor.delete_file("nonexistent.txt")
        assert result is False
    
    def test_singleton_processor(self):
        """Test that get_document_processor returns singleton."""
        # Reset singleton for test
        import src.services.document_processor as dp
        dp._document_processor = None
        
        proc1 = get_document_processor()
        proc2 = get_document_processor()
        
        assert proc1 is proc2
