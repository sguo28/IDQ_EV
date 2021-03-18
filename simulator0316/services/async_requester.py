"""Asynchronous request module"""
from concurrent import futures
import itertools
import requests
import asyncio
import osrm

class AsyncRequester(object):
    """Send asynchronous requests to list of urls
    Response time may be limited by rate of system/NW"""
    def __init__(self, n_threads=8):
        # self.urllist = []
        self.n_threads = n_threads
        self.executor = futures.ThreadPoolExecutor(max_workers=self.n_threads)
        self.client=osrm.Client(host='http://localhost:5000')
            
    async def async_requests(self,pts):
        Client=osrm.AioHTTPClient(host='http://localhost:5000')
        response=await asyncio.gather(*[
                asyncio.ensure_future(Client.nearest(coordinates=[p]))
                for p in pts
            ])
        await Client.close()
        return response
    
    def combine_async(self,pts):
        loop = asyncio.get_event_loop()
        results=loop.run_until_complete(self.async_requests(pts))
        return results

    def send_async_requests(self, pts):
        """Sends asynchronous requests"""
        if len(pts) == 1:
            return self.get_batch(pts)
        # Num of batches per thread
        n_batches = int(len(pts) / self.n_threads) + 1
        # List of URLs per batch
        batch_urllist = [pts[i * n_batches : (i + 1) * n_batches]
                        for i in range(self.n_threads)]
        # List of HTTP response
        responses = list(self.executor.map(self.get_batch, batch_urllist))
        return list(itertools.chain(*responses))    # Return 1 sequence of responses

    def get_json(self, url):
        """open URL and return JSON contents"""
#         result = requests.get(url,timeout=1).json()
        result= self.client.nearest(coordinates=[(url[1],url[0])])
        return result

    def get_batch(self, urllist):
        """Batch processing for get method; takes list of urls as input"""
        return [self.get_json(url) for url in urllist]
