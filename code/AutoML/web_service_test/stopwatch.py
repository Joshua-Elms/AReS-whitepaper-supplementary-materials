from time import sleep, perf_counter
from datetime import timedelta
from pathlib import Path
import requests
from json import loads
from math import ceil
import logging

def main(request_id, request_interval_seconds, request_timeout_seconds):
    """
    Run this program determine how long the web service takes to serve a particular request.
    """
    start = perf_counter()
    logging.info("Starting stopwatch")
    request = f"https://dalkilic.luddy.indiana.edu/api/getAutoMLRequest?request_id={request_id}"
    retries = ceil(request_timeout_seconds / request_interval_seconds)
    current_retries = 0
    job_incomplete = True
    
    while job_incomplete:

        # try request
        response = requests.get(request, timeout=request_timeout_seconds)
            
        # if request fails, increment count and try again until timeout
        if response.status_code != 200:
            logging.info("Request failed")
            current_retries += 1
            if current_retries > retries:
                logging.info("Request timed out")
                job_incomplete = False
            
        # if request succeeds, check status
        status = loads(response.content)["data"]["request_status"]
        
        # if status is 1, job is complete
        if status == "1":
            logging.info("Job completed")
            job_incomplete = False
        
        # if status is 0, job is incomplete
        else:
            logging.info("Job not completed")
            sleep(request_interval_seconds)
            current_retries = 0
            
    stop = perf_counter()
    duration = timedelta(seconds=stop-start)
    logging.info(f"Stopping stopwatch with time: {duration}")
    
    return duration
    
if __name__=="__main__":
    # req_id1 = 97febcd0-3c94-4e55-8f06-25396577f4fb
    # req_id2 = 5f2d283a-2e94-4893-895a-e4468f51156e
    logging.basicConfig(format='%(asctime)s - %(message)s', filename=Path(__file__).parent / "stopwatch.log", level=logging.INFO)
    main(
        request_id = "97febcd0-3c94-4e55-8f06-25396577f4fb",
        request_interval_seconds = 30,
        request_timeout_seconds = 300,
    )