import os
import pickle
import asyncio
from typing import List
from queue import Queue
import threading


try:
    from googleapiclient.discovery import Resource, build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
except ImportError:
    print("googleapiclient not found, please install it with `pip install google-api-python-client`")
    raise





SCOPES = ['https://mail.google.com/']


class GmailServicePool:
    """Pool of Gmail service objects for concurrent access."""
    
    def __init__(self, pool_size: int = 5):
        self.pool_size = pool_size
        self.services: Queue[Resource] = Queue()
        self._lock = threading.Lock()
        self._initialized = False
    
    def _create_service(self) -> Resource:
        """Create a single Gmail service instance."""
        secrets_dir = os.path.join(os.environ["CODEMEM_DIR"], "secrets")
        creds = None
        token_path = os.path.join(secrets_dir, "token.pickle")
        credentials_path = os.path.join(secrets_dir, "credentials.json")
        
        # Load existing token
        if os.path.exists(token_path):
            with open(token_path, "rb") as token:
                creds = pickle.load(token)
        
        # Refresh if needed
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials
            with open(token_path, "wb") as token:
                pickle.dump(creds, token)
        
        return build('gmail', 'v1', credentials=creds)
    
    def initialize(self):
        """Initialize the service pool."""
        with self._lock:
            if self._initialized:
                return
            
            print(f"🔧 Initializing Gmail service pool with {self.pool_size} connections...")
            
            for i in range(self.pool_size):
                try:
                    service = self._create_service()
                    self.services.put(service)
                    print(f"  ✅ Created service {i+1}/{self.pool_size}")
                except Exception as e:
                    print(f"  ❌ Failed to create service {i+1}: {e}")
                    # Continue with fewer services if some fail
            
            self._initialized = True
            print(f"📧 Gmail service pool ready with {self.services.qsize()} services")
    
    def get_service(self) -> Resource:
        """Get a service from the pool (blocking)."""
        if not self._initialized:
            self.initialize()
        
        return self.services.get()
    
    def return_service(self, service: Resource):
        """Return a service to the pool."""
        self.services.put(service)
    
    async def get_service_async(self) -> Resource:
        """Get a service from the pool (async)."""
        if not self._initialized:
            self.initialize()
        
        # Use asyncio to avoid blocking
        return await asyncio.to_thread(self.services.get)
    
    async def return_service_async(self, service: Resource):
        """Return a service to the pool (async)."""
        await asyncio.to_thread(self.services.put, service)


# Global service pool instance
_service_pool = None


def get_service_pool(pool_size: int = 5) -> GmailServicePool:
    """Get the global Gmail service pool."""
    global _service_pool
    if _service_pool is None:
        _service_pool = GmailServicePool(pool_size)
    return _service_pool


# SETUP INSTRUCTIONS:
# Follow the steps here to get a credentials.json file: https://thepythoncode.com/article/use-gmail-api-in-python
# Then move it to cartridges/secrets/credentials.json

def authenticate_gmail_api() -> Resource:
    print("🔐 Authenticating Gmail API...")
    
    # Check for CARTRIDGES_DIR environment variable
    if "CARTRIDGES_DIR" not in os.environ:
        raise ValueError("CARTRIDGES_DIR environment variable not set")
    
    secrets_dir = os.path.join(os.environ["CARTRIDGES_DIR"], "secrets")
    print(f"📁 Looking for credentials in: {secrets_dir}")
    
    if not os.path.exists(secrets_dir):
        raise FileNotFoundError(f"Secrets directory not found: {secrets_dir}")
    
    creds = None
    token_path = os.path.join(secrets_dir, "token.pickle")
    credentials_path = os.path.join(secrets_dir, "credentials.json")
    
    # Check if credentials.json exists
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"credentials.json not found at: {credentials_path}")
    
    # the file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time
    if os.path.exists(token_path):
        print("📄 Loading existing token...")
        try:
            with open(token_path, "rb") as token:
                creds = pickle.load(token)
        except Exception as e:
            print(f"⚠️  Failed to load token: {e}")
            creds = None
    
    # if there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        print("🔄 Refreshing or creating new credentials...")
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired token...")
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"❌ Token refresh failed: {e}")
                print("🔄 Starting new OAuth flow...")
                creds = None
        
        if not creds:
            print("🌐 Starting OAuth flow...")
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, 
                    SCOPES
                )
                creds = flow.run_local_server(port=0)
            except Exception as e:
                print(f"❌ OAuth flow failed: {e}")
                raise
        
        # save the credentials for the next run
        print("💾 Saving credentials...")
        try:
            with open(token_path, "wb") as token:
                pickle.dump(creds, token)
        except Exception as e:
            print(f"⚠️  Failed to save token: {e}")
    
    print("✅ Gmail API authentication successful")
    try:
        service = build('gmail', 'v1', credentials=creds)
        # Test the connection with a simple API call
        print("🧪 Testing Gmail API connection...")
        profile = service.users().getProfile(userId='me').execute()
        print(f"📧 Connected to Gmail account: {profile.get('emailAddress', 'Unknown')}")
        return service
    except Exception as e:
        print(f"❌ Gmail API connection test failed: {e}")
        raise
