import os
import torch
import runpod
import logging
import asyncio
import uuid
import numpy as np
import time
import math
from typing import Dict, Any, List, Optional, Union
from collections import deque
import threading

from binoculars.detector import Binoculars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

binoculars = None
batch_processor = None
INITIALIZATION_COMPLETE = False

# Environment variables
MODE = os.environ.get("BINOCULARS_MODE", "low-fpr")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 512))
USE_BFLOAT16 = os.environ.get("USE_BFLOAT16", "true").lower() == "true"
COMPILE = os.environ.get("COMPILE", "false").lower() == "true"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
MAX_WAIT_TIME = float(os.environ.get("MAX_WAIT_TIME", 0.05))

def random_uuid():
    return str(uuid.uuid4())

def safe_float(value, default=0.0):
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            logger.warning(f"Invalid score {f}, replacing with {default}")
            return default
        return f
    except (ValueError, TypeError):
        return default

class JobInput:
    def __init__(self, job):
        logger.debug(f"Received job structure: {job}")
        
        self.text = job.get("text", "")
        if not isinstance(self.text, str):
            logger.error(f"Text must be string, got {type(self.text)}: {self.text}")
            raise ValueError(f"Text must be string, got {type(self.text)}")
        
        if not self.text.strip():
            logger.warning(f"Empty or whitespace-only text received: '{self.text}'")
        
        self.request_id = job.get("request_id") or random_uuid()

class IntelligentBatchProcessor:
    """
    Intelligent batch processor for RunPod concurrent handlers
    Accumulates requests and processes them in optimal batches using Binoculars
    """
    def __init__(self, binoculars: Binoculars, batch_size: int = 8, max_wait_time: float = 0.05):
        self.binoculars = binoculars
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = deque()
        self.processing_lock = asyncio.Lock()
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'failed_requests': 0
        }
        
        logger.info(f"IntelligentBatchProcessor initialized with batch_size={batch_size}")
    
    async def process_request(self, text: str, request_id: Optional[str] = None) -> float:
        """Process a single request with intelligent batching - returns only the score"""
        
        # Create future for this request
        future = asyncio.Future()
        request = {
            'text': text,
            'request_id': request_id,
            'future': future,
            'timestamp': time.time()
        }
        
        # Add to pending requests
        self.pending_requests.append(request)
        
        # Trigger batch processing if we have enough requests
        if len(self.pending_requests) >= self.batch_size:
            asyncio.create_task(self._process_batch())
        else:
            # Set a timeout to process smaller batches
            asyncio.create_task(self._process_after_timeout())
        
        # Wait for result
        try:
            score = await asyncio.wait_for(future, timeout=30)
            return score
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out")
            return 0.0
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            return 0.0
    
    async def _process_after_timeout(self):
        """Process pending requests after timeout"""
        await asyncio.sleep(self.max_wait_time)
        await self._process_batch()
    
    async def _process_batch(self):
        """Process batch of requests using Binoculars.compute_score with list of texts"""
        async with self.processing_lock:
            if not self.pending_requests:
                return
            
            # Extract batch
            batch_requests = []
            for _ in range(min(self.batch_size, len(self.pending_requests))):
                if self.pending_requests:
                    batch_requests.append(self.pending_requests.popleft())
            
            if not batch_requests:
                return
            
            try:
                # Extract texts for batch processing
                texts = [req['text'] for req in batch_requests]
                
                batch_start_time = time.time()
                logger.info(f"🚀 Processing batch of {len(texts)} requests")
                
                loop = asyncio.get_event_loop()
                scores = await loop.run_in_executor(None, self.binoculars.compute_score, texts)
                
                if not isinstance(scores, list):
                    scores = [scores]  # Handle edge case
                
                # Validate scores count
                if len(scores) != len(batch_requests):
                    raise ValueError(f"Score count mismatch: {len(scores)} scores for {len(batch_requests)} requests")
                
                # Sanitize and set results for all requests
                for request, score in zip(batch_requests, scores):
                    if not request['future'].done():
                        safe_score = safe_float(score, 0.0)
                        request['future'].set_result(safe_score)
                
                # Update statistics
                batch_time = time.time() - batch_start_time
                self.stats['total_batches'] += 1
                self.stats['total_requests'] += len(batch_requests)
                self.stats['avg_batch_size'] = self.stats['total_requests'] / self.stats['total_batches']
                
                logger.info(f"✅ Batch of {len(batch_requests)} requests completed in {batch_time:.3f}s")
                
            except Exception as e:
                logger.error(f"❌ Batch processing error: {e}")
                
                # Set default scores for all requests in the batch
                self.stats['failed_requests'] += len(batch_requests)
                
                for request in batch_requests:
                    if not request['future'].done():
                        request['future'].set_result(0.0)  # Default score on error
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            **self.stats,
            'pending_requests': len(self.pending_requests)
        }

def init_binoculars():
    """Initialize Binoculars models with optimizations"""
    global binoculars, batch_processor, INITIALIZATION_COMPLETE
    
    try:
        logger.info("Starting Binoculars initialization...")
        
        if binoculars is None:
            logger.info("Loading Binoculars models...")
            
            torch.set_float32_matmul_precision("medium")
            
            binoculars = Binoculars(
                observer_name_or_path=os.environ.get("OBSERVER_MODEL", "tiiuae/falcon-7b"),
                performer_name_or_path=os.environ.get("PERFORMER_MODEL", "tiiuae/falcon-7b-instruct"),
                use_bfloat16=USE_BFLOAT16,
                max_token_observed=MAX_TOKENS,
                mode=MODE,
                check_tokenizer_consistency=False,
                compile=COMPILE,
            )
            
            logger.info("Binoculars models loaded successfully")
        
        if batch_processor is None:
            logger.info("Initializing batch processor...")
            batch_processor = IntelligentBatchProcessor(binoculars, BATCH_SIZE, MAX_WAIT_TIME)
            logger.info("Batch processor initialized successfully")
        
        # Test Binoculars batching
        logger.info("Testing Binoculars functionality...")
        test_scores = binoculars.compute_score(["Test text 1", "Test text 2"])
        logger.info(f"Binoculars batch test: {test_scores} (type: {type(test_scores)})")
        
        INITIALIZATION_COMPLETE = True
        logger.info("✅ Complete initialization finished successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Binoculars: {e}")
        INITIALIZATION_COMPLETE = False
        raise
    
    return binoculars, batch_processor

async def handler(job: Dict[str, Any]) -> float:
    """
    RunPod concurrent handler with intelligent batching
    Returns only the score value (float)
    """
    try:
        if not INITIALIZATION_COMPLETE or batch_processor is None:
            logger.error("Batch processor not initialized or initialization incomplete")
            return 0.0
        
        # Parse job input
        job_input = JobInput(job["input"])
        
        if not job_input.text or not job_input.text.strip():
            logger.error(f"Empty or whitespace-only input text received: '{job_input.text}'")
            return 0.0
        
        # Process through batch processor - returns only score
        score = await batch_processor.process_request(
            job_input.text,
            job_input.request_id
        )
        
        logger.info(f"Request {job_input.request_id} completed: score={score:.4f}")
        return score
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return 0.0

def concurrency_modifier(current_concurrency: int) -> int:
    """
    Return batch size to ensure optimal batching
    """
    try:
        if not INITIALIZATION_COMPLETE or batch_processor is None:
            logger.warning("Batch processor not initialized, using default concurrency")
            return BATCH_SIZE
            
        stats = batch_processor.get_stats()
        pending = stats.get('pending_requests', 0)
        avg_batch_size = stats.get('avg_batch_size', 1.0)
        
        # Log batching efficiency
        logger.debug(f"Batching stats: avg_batch_size={avg_batch_size:.2f}, pending={pending}")
        
        # Return batch size to encourage full batches
        return BATCH_SIZE
        
    except Exception as e:
        logger.warning(f"Concurrency modifier error: {e}")
        return BATCH_SIZE

def health_check() -> Dict[str, Any]:
    """Health check with batch processor stats"""
    try:
        stats = batch_processor.get_stats() if (INITIALIZATION_COMPLETE and batch_processor) else {}
        return {
            "status": "healthy" if INITIALIZATION_COMPLETE else "initializing",
            "initialization_complete": INITIALIZATION_COMPLETE,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
            "models_loaded": binoculars is not None,
            "batch_processor_loaded": batch_processor is not None,
            "batch_processor_stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "initialization_complete": INITIALIZATION_COMPLETE
        }

if __name__ == "__main__":
    logger.info("🚀 Starting RunPod Binoculars worker with intelligent batching...")
    
    # Konfiguracja
    logger.info(f"Configuration:")
    logger.info(f"  - Batch size: {BATCH_SIZE}")
    logger.info(f"  - Max wait time: {MAX_WAIT_TIME}s")
    logger.info(f"  - GPU devices: {torch.cuda.device_count()}")
    logger.info(f"  - Use bfloat16: {USE_BFLOAT16}")
    logger.info(f"  - Compile: {COMPILE}")
    logger.info(f"  - Mode: {MODE}")
    logger.info(f"  - Max tokens: {MAX_TOKENS}")

    try:
        logger.info("🔄 Initializing models...")
        binoculars, batch_processor = init_binoculars()
        logger.info("✅ Models initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        logger.error("Exiting due to initialization failure")
        exit(1)
    

    health = health_check()
    logger.info(f"Initial health check: {health}")
    
    if not health.get("initialization_complete", False):
        logger.error("❌ Initialization not complete, exiting")
        exit(1)
    
    try:
        logger.info("🎯 Starting RunPod serverless worker...")
        # Start RunPod serverless worker
        runpod.serverless.start({
            "handler": handler,
            "concurrency_modifier": concurrency_modifier,
        })
    except KeyboardInterrupt:
        logger.info("📡 Received shutdown signal")
    except Exception as e:
        logger.error(f"❌ Worker startup error: {e}")
        raise
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🏁 Worker shutdown complete")
