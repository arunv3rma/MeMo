
import os
import random
import asyncio
import math
from typing import List, Optional, Dict
from dataclasses import dataclass
from pydantic import BaseModel
from datetime import datetime, timedelta
from collections import defaultdict

from cartridges.data.resources import Resource
from .utils import get_service_pool

class LabelConfig(BaseModel):
    name: str
    weight: float = 1.0
    

@dataclass
class ThreadMetadata:
    id: str
    label: str
    date_bucket: str  # Format: "2025-06-01_to_2025-06-30"
    weight: Optional[float] = None


class GmailResource(Resource):

    class Config(Resource.Config):

        # note: use the label as it appears in gmail search query (not the sidebar)
        labels: Optional[List[LabelConfig]]=None

        # date format is YYYY/MM/DD
        date_start: Optional[str] = "2025/01/01"
        date_end: Optional[str] = None
        date_days_in_bucket: int = 30
        
        # Exponential decay parameters for temporal bias
        temporal_decay_rate: float = 0.1  # Higher = stronger recency bias
        temporal_half_life_days: Optional[int] = None  # Alternative to decay_rate
        

    def __init__(self, config: Config):
        self.config = config
        # Get service pool for concurrent access
        self.service_pool = get_service_pool(pool_size=16)
        # Limit concurrent API calls to avoid rate limits
        self._api_semaphore = asyncio.Semaphore(16)  # Max 5 concurrent calls
    

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        """Sample a Gmail thread and return prompts for synthesis."""
        if not self.threads:
            raise ValueError("No threads loaded. Call setup() first.")
        
        # Sample a random thread
        sampled_thread = await self._sample_thread()
        
        # Get the thread content
        thread_content = await self._get_thread_content(sampled_thread.id)
        
        # Sample prompts from the predefined list
        prompts = random.choices(THREAD_PROMPTS, k=batch_size)
        
        return thread_content, prompts


    async def setup(self):
        """
        Fetch the metadata (ids) for all the threads in the user's gmail with 
        the labels specified in the config. Uses separate queries per label and date bucket.
        """
        print("🔄 Setting up Gmail resource...")
        self.threads: List[ThreadMetadata] = []
        self.threads_by_bucket: Dict[str, List[ThreadMetadata]] = {}
        self.threads_by_label: Dict[str, Dict[str, List[ThreadMetadata]]] = {}
        
        try:
            # Generate date buckets
            date_buckets = self._generate_date_buckets()
            
            # Create all fetch tasks concurrently
            fetch_tasks = []
            
            if not self.config.labels:
                # No labels specified, fetch all threads for each date bucket
                for bucket_start, bucket_end, bucket_name in date_buckets:
                    task = self._fetch_threads_for_label_and_date(None, 1.0, bucket_start, bucket_end, bucket_name)
                    fetch_tasks.append(task)
            else:
                # Fetch threads for each label and date bucket combination
                for label_config in self.config.labels:
                    for bucket_start, bucket_end, bucket_name in date_buckets:
                        task = self._fetch_threads_for_label_and_date(
                            label_config.name, label_config.weight, 
                            bucket_start, bucket_end, bucket_name
                        )
                        fetch_tasks.append(task)
            
            # Execute all fetch operations concurrently and collect results
            print(f"🚀 Starting {len(fetch_tasks)} concurrent fetch operations...")
            task_results = await asyncio.gather(*fetch_tasks)
            
            # Merge all results into main thread list
            for thread_list in task_results:
                if thread_list:  # Skip empty results
                    self.threads.extend(thread_list)
            
            print(f"📊 Collected {len(self.threads)} threads from concurrent operations")
            
            # Organize threads by bucket for efficient sampling
            self._organize_threads_by_bucket()
            
            print(f"✅ Loaded {len(self.threads)} Gmail threads across {len(date_buckets)} date buckets")
                
        except Exception as e:
            print(f"❌ Error setting up Gmail resource: {e}")
            raise
    
    def _generate_date_buckets(self) -> List[tuple]:
        """Generate date buckets based on config."""
        buckets = []
        
        if not self.config.date_start:
            # No date range specified, create a single "all time" bucket
            return [(None, None, "all_time")]
        
        start_date = datetime.strptime(self.config.date_start, "%Y/%m/%d")
        
        # If date_end is not provided, use today's date
        if self.config.date_end:
            end_date = datetime.strptime(self.config.date_end, "%Y/%m/%d")
        else:
            end_date = datetime.now().replace(hour=23, minute=59, second=59)  # End of today
        
        bucket_days = self.config.date_days_in_bucket
        
        current_date = start_date
        while current_date < end_date:
            bucket_end = min(current_date + timedelta(days=bucket_days), end_date)
            
            # Format dates for Gmail query (YYYY/MM/DD)
            bucket_start_str = current_date.strftime("%Y/%m/%d")
            bucket_end_str = bucket_end.strftime("%Y/%m/%d")
            
            # Create bucket name
            bucket_name = f"{current_date.strftime('%Y-%m-%d')}_to_{bucket_end.strftime('%Y-%m-%d')}"
            
            buckets.append((bucket_start_str, bucket_end_str, bucket_name))
            current_date = bucket_end
        
        print(f"📅 Generated {len(buckets)} date buckets of {bucket_days} days each")
        return buckets
    
    async def _fetch_threads_for_label_and_date(self, label_name: Optional[str], weight: float, 
                                               date_start: Optional[str], date_end: Optional[str], 
                                               bucket_name: str) -> List[ThreadMetadata]:
        """Fetch all threads for a specific label and date range. Returns list of threads."""
        query_parts = []
        
        if label_name:
            query_parts.append(f"label:{label_name}")
        
        if date_start:
            query_parts.append(f"after:{date_start}")
        
        if date_end:
            query_parts.append(f"before:{date_end}")
        
        query = " ".join(query_parts) if query_parts else ""
        
        label_display = label_name if label_name else "ALL"
        print(f"📧 Fetching threads for {label_display} in bucket {bucket_name}")
        
        page_token = None
        bucket_thread_count = 0
        collected_threads = []  # Local collection for this task
        
        while True:
            # Use minimal fields to improve performance
            request_params = {
                'q': query,
                'fields': 'threads(id),nextPageToken',
                'maxResults': 500  # Maximum allowed by API
            }
            
            if page_token:
                request_params['pageToken'] = page_token
            
            # Get service from pool and fetch thread list
            async with self._api_semaphore:
                service = await self.service_pool.get_service_async()
                try:
                    result = await asyncio.to_thread(
                        lambda: service.users().threads().list(
                            userId='me',
                            **request_params
                        ).execute()
                    )
                finally:
                    await self.service_pool.return_service_async(service)
            
            threads = result.get('threads', [])
            
            if threads:
                for thread in threads:
                    thread_id = thread['id']
                    
                    # Create thread metadata with known label and date bucket
                    thread_metadata = ThreadMetadata(
                        id=thread_id,
                        label=label_name or "ALL",
                        date_bucket=bucket_name,
                        weight=weight
                    )
                    
                    # Collect in local list (thread-safe)
                    collected_threads.append(thread_metadata)
                    bucket_thread_count += 1
                    
                    # Progress logging
                    if bucket_thread_count % 100 == 0:
                        print(f"  📊 {label_display}/{bucket_name}: {bucket_thread_count} threads...")
            
            # Check for next page
            page_token = result.get('nextPageToken')
            if not page_token:
                break
        
        print(f"  ✅ {label_display}/{bucket_name}: {bucket_thread_count} threads loaded")
        return collected_threads
  
    def _is_more_recent_bucket(self, bucket_a: str, bucket_b: str) -> bool:
        """Check if bucket_a is more recent than bucket_b."""
        if bucket_a == "all_time" or bucket_b == "all_time":
            return False  # Can't compare with all_time bucket
        
        try:
            # Extract start dates from bucket names (format: "2025-06-01_to_2025-06-30")
            date_a = bucket_a.split("_to_")[0]
            date_b = bucket_b.split("_to_")[0]
            return date_a > date_b  # String comparison works for YYYY-MM-DD format
        except:
            return False
    
    def _organize_threads_by_bucket(self):
        """Organize threads by label and date bucket for efficient sampling."""
        # Organize by both label and date bucket
        for thread in self.threads:
            label = thread.label
            bucket = thread.date_bucket
            
            # Add to bucket-based organization
            if bucket not in self.threads_by_bucket:
                self.threads_by_bucket[bucket] = []
            self.threads_by_bucket[bucket].append(thread)
            
            # Add to label-based organization
            if label not in self.threads_by_label:
                self.threads_by_label[label] = {}
            if bucket not in self.threads_by_label[label]:
                self.threads_by_label[label][bucket] = []
            self.threads_by_label[label][bucket].append(thread)
        
        # Print statistics
        print("📊 Thread organization:")
        for label in sorted(self.threads_by_label.keys()):
            total_in_label = sum(len(buckets) for buckets in self.threads_by_label[label].values())
            print(f"  🏷️  Label '{label}': {total_in_label} threads")
            for bucket in sorted(self.threads_by_label[label].keys(), reverse=True):
                count = len(self.threads_by_label[label][bucket])
                print(f"    📅 {bucket}: {count} threads")
        
        print(f"📊 Organized {len(self.threads)} threads into {len(self.threads_by_label)} labels × {len(self.threads_by_bucket)} date buckets")

    
    async def _sample_thread(self) -> ThreadMetadata:
        """Sample a random thread with label and recency bias applied at sampling time."""
        if not self.threads_by_label:
            raise ValueError("No threads loaded. Call setup() first.")
        
        # (1) Sample a label with weight bias
        label = self._sample_label()
        
        # (2) Sample a date bucket with recency bias for that label
        bucket = self._sample_date_bucket_for_label(label)
        
        # (3) Sample a thread from that label×bucket combination
        bucket_threads = self.threads_by_label[label][bucket]
        return random.choice(bucket_threads)
    
    def _sample_label(self) -> str:
        """Sample a label based on configured weights."""
        if not self.config.labels:
            # No label configuration, return first available label
            return list(self.threads_by_label.keys())[0]
        
        # Create weighted list of labels
        labels = []
        weights = []
        
        for label_config in self.config.labels:
            if label_config.name in self.threads_by_label:
                labels.append(label_config.name)
                weights.append(label_config.weight)
        
        if not labels:
            # Fallback to any available label
            return list(self.threads_by_label.keys())[0]
        
        # Use weighted random selection
        return random.choices(labels, weights=weights)[0]
    
    def _sample_date_bucket_for_label(self, label: str) -> str:
        """Sample a date bucket with exponential temporal decay for a specific label."""
        if label not in self.threads_by_label:
            raise ValueError(f"Label '{label}' not found in threads")
        
        buckets = list(self.threads_by_label[label].keys())
        
        if len(buckets) == 1:
            return buckets[0]
        
        # Calculate temporal weights using exponential decay
        weights = []
        current_time = datetime.now()
        
        for bucket in buckets:
            bucket_size = len(self.threads_by_label[label][bucket])
            
            # Calculate days since bucket start date
            days_ago = self._get_days_since_bucket(bucket, current_time)
            
            # Calculate exponential decay weight
            temporal_weight = self._calculate_temporal_weight(days_ago)
            
            # Combine temporal weight with bucket size (dampened)
            combined_weight = temporal_weight * (bucket_size ** 0.3)  # Light bucket size influence
            weights.append(combined_weight)
        
        # Use weighted random selection
        return random.choices(buckets, weights=weights)[0]
    
    def _get_days_since_bucket(self, bucket_name: str, current_time: datetime) -> float:
        """Calculate days since the start of a date bucket."""
        if bucket_name == "all_time":
            return 365.0  # Treat as very old
        
        try:
            # Extract start date from bucket name (format: "2025-06-01_to_2025-06-30")
            bucket_start_str = bucket_name.split("_to_")[0]
            bucket_start = datetime.strptime(bucket_start_str, "%Y-%m-%d")
            days_ago = (current_time - bucket_start).total_seconds() / (24 * 3600)
            return max(0, days_ago)  # Ensure non-negative
        except (ValueError, IndexError):
            return 365.0  # Default to old if parsing fails
    
    def _calculate_temporal_weight(self, days_ago: float) -> float:
        """Calculate exponential decay weight based on days ago."""
        if self.config.temporal_half_life_days is not None:
            # Use half-life approach: weight = exp(-ln(2) * days_ago / half_life)
            decay_constant = math.log(2) / self.config.temporal_half_life_days
            return math.exp(-decay_constant * days_ago)
        else:
            # Use decay rate approach: weight = exp(-decay_rate * days_ago)
            return math.exp(-self.config.temporal_decay_rate * days_ago)

    async def _get_thread_content(self, thread_id: str) -> str:
        """Get the full content of a Gmail thread using minimal fields."""
        try:
            # Get service from pool and fetch thread content
            async with self._api_semaphore:
                service = await self.service_pool.get_service_async()
                try:
                    thread = await asyncio.to_thread(
                        lambda: service.users().threads().get(
                            userId='me',
                            id=thread_id,
                            fields='messages(payload(headers,parts,body),snippet)'
                        ).execute()
                    )
                finally:
                    await self.service_pool.return_service_async(service)
            
            messages = thread.get('messages', [])
            thread_content = []
            
            for message in messages:
                # Extract message content
                payload = message.get('payload', {})
                
                # Get headers for subject and sender
                headers = payload.get('headers', [])
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                from_addr = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
                
                # Get message body
                body = self._extract_message_body(payload)
                
                thread_content.append(f"From: {from_addr}\nSubject: {subject}\n\n{body}\n" + "="*50)
            
            return "\n\n".join(thread_content)
            
        except Exception as e:
            return f"Error loading thread {thread_id}: {str(e)}"
    
    def _extract_message_body(self, payload: dict) -> str:
        """Extract text from Gmail message payload."""
        body = ""
        
        # Handle different payload structures
        if 'parts' in payload:
            # Multi-part message
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain':
                    part_body = part.get('body', {}).get('data', '')
                    if part_body:
                        # Decode base64url
                        import base64
                        decoded = base64.urlsafe_b64decode(part_body + '==').decode('utf-8', errors='ignore')
                        body += decoded
        else:
            # Single part message
            if payload.get('mimeType') == 'text/plain':
                body_data = payload.get('body', {}).get('data', '')
                if body_data:
                    import base64
                    body = base64.urlsafe_b64decode(body_data + '==').decode('utf-8', errors='ignore')
        
        return body.strip() if body else "No readable content"

   

THREAD_PROMPTS = [
    (
        "You are analyzing an email thread. Please identify a key insight, decision, or important information shared in the conversation and generate a question that can be used to test understanding of this insight."
        "Be sure to include specific details (names, dates, concepts, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),

    (
        "You are analyzing an email thread. Please identify something the user learned or discovered from the conversation and generate a question that can be used to test understanding of this lesson."
        "Be sure to include specific details (names, dates, concepts, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),

    (
        "You are analyzing an email thread. Please identify a key challenge, problem, or difficulty discussed in the conversation and generate a question that tests understanding of this issue."
        "Be sure to include specific details (names, dates, concepts, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),

    (
        "You are analyzing an email thread. Please identify the main topic or subject of the conversation and generate a comprehensive question that tests understanding of the discussion."
        "Be sure to include specific details (names, dates, concepts, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),

    (
        "You are analyzing an email thread. Please generate a question that tests understanding of the overall context and outcome of the email conversation."
        "Be sure to include specific details (names, dates, concepts, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),
    (
        "You are analyzing an email thread. Please generate a single instruction for an LLM to summarize the most important part of this email conversation."
    )
]


