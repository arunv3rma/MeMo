import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from cartridges.data.resources import BaseStructuredResource, DirectoryResource


class TestStructuredResource(BaseStructuredResource):
    """Test implementation of BaseStructuredResource"""
    
    def __init__(self, config, test_data):
        self.test_data = test_data
        super().__init__(config)
    
    def _load_data(self):
        return self.test_data


class TestListNestedData:
    """Test cases for the _list_nested_data method"""
    
    def test_simple_dict_leaves_only_true(self):
        """Test simple dictionary with leaves_only=True"""
        config = Mock()
        config.leaves_only = True
        config.seed_prompts = []
        
        test_data = {"a": 1, "b": "hello", "c": True}
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [("a", "1"), ("b", "hello"), ("c", "True")]
        
        assert sorted(result) == sorted(expected)
    
    def test_simple_dict_leaves_only_false(self):
        """Test simple dictionary with leaves_only=False"""
        config = Mock()
        config.leaves_only = False
        config.seed_prompts = []
        
        test_data = {"a": 1, "b": "hello"}
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        
        # Should include the dict itself plus leaf values
        assert ("", str(test_data)) in result
        assert ("a", "1") in result
        assert ("b", "hello") in result
        assert len(result) == 3
    
    def test_nested_dict_leaves_only_true(self):
        """Test nested dictionary with leaves_only=True"""
        config = Mock()
        config.leaves_only = True
        config.seed_prompts = []
        
        test_data = {
            "user": {
                "name": "John",
                "age": 30,
                "details": {
                    "city": "NYC"
                }
            },
            "status": "active"
        }
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [
            ("user/name", "John"),
            ("user/age", "30"),
            ("user/details/city", "NYC"),
            ("status", "active")
        ]
        
        assert sorted(result) == sorted(expected)
    
    def test_nested_dict_leaves_only_false(self):
        """Test nested dictionary with leaves_only=False"""
        config = Mock()
        config.leaves_only = False
        config.seed_prompts = []
        
        test_data = {
            "user": {
                "name": "John",
                "age": 30
            }
        }
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        
        # Should include all levels: root dict, nested dict, and leaf values
        paths = [item[0] for item in result]
        assert "" in paths  # root dict
        assert "user" in paths  # nested dict
        assert "user/name" in paths  # leaf value
        assert "user/age" in paths  # leaf value
    
    def test_simple_list_leaves_only_true(self):
        """Test simple list with leaves_only=True"""
        config = Mock()
        config.leaves_only = True
        config.seed_prompts = []
        
        test_data = [1, "hello", True]
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [("0", "1"), ("1", "hello"), ("2", "True")]
        
        assert sorted(result) == sorted(expected)
    
    def test_simple_list_leaves_only_false(self):
        """Test simple list with leaves_only=False"""
        config = Mock()
        config.leaves_only = False
        config.seed_prompts = []
        
        test_data = [1, "hello"]
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        
        # Should include the list itself plus leaf values
        assert ("", str(test_data)) in result
        assert ("0", "1") in result
        assert ("1", "hello") in result
        assert len(result) == 3
    
    def test_nested_list_leaves_only_true(self):
        """Test nested list with leaves_only=True"""
        config = Mock()
        config.leaves_only = True
        config.seed_prompts = []
        
        test_data = [1, [2, 3], [4, [5, 6]]]
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [
            ("0", "1"),
            ("1/0", "2"),
            ("1/1", "3"),
            ("2/0", "4"),
            ("2/1/0", "5"),
            ("2/1/1", "6")
        ]
        
        assert sorted(result) == sorted(expected)
    
    def test_mixed_dict_list_leaves_only_true(self):
        """Test mixed dictionary and list structures with leaves_only=True"""
        config = Mock()
        config.leaves_only = True
        config.seed_prompts = []
        
        test_data = {
            "items": [
                {"name": "item1", "value": 10},
                {"name": "item2", "value": 20}
            ],
            "metadata": {
                "total": 2,
                "tags": ["tag1", "tag2"]
            }
        }
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [
            ("items/0/name", "item1"),
            ("items/0/value", "10"),
            ("items/1/name", "item2"),
            ("items/1/value", "20"),
            ("metadata/total", "2"),
            ("metadata/tags/0", "tag1"),
            ("metadata/tags/1", "tag2")
        ]
        
        assert sorted(result) == sorted(expected)
    
    def test_mixed_dict_list_leaves_only_false(self):
        """Test mixed dictionary and list structures with leaves_only=False"""
        config = Mock()
        config.leaves_only = False
        config.seed_prompts = []
        
        test_data = {
            "items": [{"name": "item1"}],
            "count": 1
        }
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        
        # Should include containers at all levels
        paths = [item[0] for item in result]
        assert "" in paths  # root dict
        assert "items" in paths  # list
        assert "items/0" in paths  # dict in list
        assert "items/0/name" in paths  # leaf value
        assert "count" in paths  # leaf value
    
    def test_primitive_value(self):
        """Test with primitive value (not dict or list)"""
        config = Mock()
        config.leaves_only = True
        config.seed_prompts = []
        
        test_data = "simple string"
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [("", "simple string")]
        
        assert result == expected
    
    def test_empty_dict(self):
        """Test with empty dictionary"""
        config = Mock()
        config.leaves_only = False
        config.seed_prompts = []
        
        test_data = {}
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [("", "{}")]
        
        assert result == expected
    
    def test_empty_list(self):
        """Test with empty list"""
        config = Mock()
        config.leaves_only = False
        config.seed_prompts = []
        
        test_data = []
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [("", "[]")]
        
        assert result == expected


class TestDirectoryResource:
    """Test cases for DirectoryResource"""
    
    @pytest.fixture
    def temp_dir_with_files(self):
        """Create a temporary directory with test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files with different extensions
            test_files = {
                "test.py": "print('Hello, World!')",
                "config.json": '{"name": "test", "value": 42}',
                "README.txt": "This is a README file",
                "data.yaml": "key: value\nlist:\n  - item1\n  - item2",
                "binary.exe": b"\x00\x01\x02\x03",  # Binary file
                "ignored.log": "This should be ignored"
            }
            
            for filename, content in test_files.items():
                filepath = os.path.join(temp_dir, filename)
                mode = 'wb' if isinstance(content, bytes) else 'w'
                encoding = None if isinstance(content, bytes) else 'utf-8'
                with open(filepath, mode, encoding=encoding) as f:
                    f.write(content)
            
            yield temp_dir, test_files
    
    def test_init(self):
        """Test DirectoryResource initialization"""
        config = Mock()
        config.path = "/test/path"
        config.included_extensions = [".py", ".txt"]
        config.seed_prompts = ["generic"]
        
        resource = DirectoryResource(config)
        
        assert resource.config == config
        assert resource.files == []
    
    @pytest.mark.asyncio
    async def test_setup_filters_by_extension(self, temp_dir_with_files):
        """Test that setup correctly filters files by extension"""
        temp_dir, test_files = temp_dir_with_files
        
        config = Mock()
        config.path = temp_dir
        config.included_extensions = [".py", ".txt", ".json"]
        config.seed_prompts = ["generic"]
        
        resource = DirectoryResource(config)
        await resource.setup()
        
        # Should include only files with specified extensions
        expected_files = {"test.py", "config.json", "README.txt"}
        assert set(resource.files) == expected_files
    
    @pytest.mark.asyncio
    async def test_setup_empty_directory(self):
        """Test setup with empty directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Mock()
            config.path = temp_dir
            config.included_extensions = [".py", ".txt"]
            config.seed_prompts = ["generic"]
            
            resource = DirectoryResource(config)
            await resource.setup()
            
            assert resource.files == []
    
    @pytest.mark.asyncio
    async def test_sample_prompt_success(self, temp_dir_with_files):
        """Test successful sample_prompt execution"""
        temp_dir, test_files = temp_dir_with_files
        
        config = Mock()
        config.path = temp_dir
        config.included_extensions = [".py", ".txt"]
        config.seed_prompts = ["generic"]
        
        resource = DirectoryResource(config)
        await resource.setup()
        
        with patch('cartridges.data.resources.sample_seed_prompts') as mock_sample:
            mock_sample.return_value = ["test prompt 1", "test prompt 2"]
            
            context, seed_prompts = await resource.sample_prompt(2)
            
            # Should return context with file content
            assert isinstance(context, str)
            assert context.startswith("File: ")
            assert seed_prompts == ["test prompt 1", "test prompt 2"]
            mock_sample.assert_called_once_with(["generic"], 2)
    
    @pytest.mark.asyncio
    async def test_sample_prompt_no_files_error(self):
        """Test that sample_prompt raises error when no files available"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Mock()
            config.path = temp_dir
            config.included_extensions = [".py", ".txt"]
            config.seed_prompts = ["generic"]
            
            resource = DirectoryResource(config)
            await resource.setup()
            
            with pytest.raises(ValueError, match="No files found in directory"):
                await resource.sample_prompt(1)
    
    @pytest.mark.asyncio
    async def test_sample_prompt_with_specific_file(self, temp_dir_with_files):
        """Test that sample_prompt correctly reads specific file content"""
        temp_dir, test_files = temp_dir_with_files
        
        config = Mock()
        config.path = temp_dir
        config.included_extensions = [".json"]
        config.seed_prompts = ["generic"]
        
        resource = DirectoryResource(config)
        await resource.setup()
        
        # Should only have config.json
        assert resource.files == ["config.json"]
        
        with patch('cartridges.data.resources.sample_seed_prompts') as mock_sample:
            mock_sample.return_value = ["test prompt"]
            
            context, _ = await resource.sample_prompt(1)
            
            expected_content = test_files["config.json"]
            assert f"File: config.json\n\n{expected_content}" == context
    
    @pytest.mark.asyncio
    async def test_sample_prompt_encoding_fallback(self):
        """Test that sample_prompt handles encoding issues gracefully"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file with non-UTF-8 encoding
            binary_file = os.path.join(temp_dir, "binary.txt")
            with open(binary_file, 'wb') as f:
                f.write(b'\xff\xfe\x00\x01')  # Invalid UTF-8
            
            config = Mock()
            config.path = temp_dir
            config.included_extensions = [".txt"]
            config.seed_prompts = ["generic"]
            
            resource = DirectoryResource(config)
            await resource.setup()
            
            with patch('cartridges.data.resources.sample_seed_prompts') as mock_sample:
                mock_sample.return_value = ["test prompt"]
                
                context, _ = await resource.sample_prompt(1)
                
                # Should handle encoding gracefully
                assert "File: binary.txt" in context
                assert isinstance(context, str)
