from concurrent.futures import ThreadPoolExecutor
import threading
import pytest
from egoannot.quality.shared_hand import SharedHandExecutor


class Detector:
    model_name='fake'
    def __init__(self):
        self.active=0
        self.peak=0
    def __call__(self, frame, offset=(0,0), source='full'):
        self.active+=1
        self.peak=max(self.peak,self.active)
        try:
            if frame=='fail':
                raise ValueError('Bad image')
            return [dict(frame=frame,offset=offset,source=source)]
        finally:
            self.active-=1


def test_requests_keep_episode_identity_and_counters():
    detector=Detector(); server=SharedHandExecutor(detector)
    a,b=server.client(),server.client()
    try:
        with ThreadPoolExecutor(2) as pool:
            x=pool.submit(a,'a',(4,5),'crop')
            y=pool.submit(b,'b')
            assert x.result()==[dict(frame='a',offset=(4,5),source='crop')]
            assert y.result()==[dict(frame='b',offset=(0,0),source='full')]
        assert a.calls==b.calls==1
        assert detector.peak==1
    finally:
        server.close()
    assert not server.thread.is_alive()
    with pytest.raises(RuntimeError,match='closed'):
        a('later')


def test_failure_propagates_without_killing_other_episodes():
    server=SharedHandExecutor(Detector())
    try:
        with pytest.raises(ValueError,match='Bad image'):
            server.client()('fail')
        assert server.client()('good')[0]['frame']=='good'
    finally:
        server.close()


def test_batch_preserves_input_output_order(monkeypatch):
    from egoannot.quality import shared_hand
    observed=[]
    def batch(detector,requests):
        observed.append(requests)
        return [[request[0]] for request in requests]
    monkeypatch.setattr(shared_hand,'owl_batch',batch)
    server=SharedHandExecutor(Detector(),max_batch=2,wait_ms=500)
    barrier=threading.Barrier(2)
    def one(value):
        client=server.client();barrier.wait()
        return client(value)
    try:
        with ThreadPoolExecutor(2) as pool:
            assert list(pool.map(one,['a','b']))==[['a'],['b']]
        assert len(observed)==1 and len(observed[0])==2
    finally:
        server.close()


def test_close_flushes_pending_request():
    server=SharedHandExecutor(Detector(),max_batch=4,wait_ms=500)
    accepted=threading.Event()
    put=server.queue.put
    def signal(item):
        put(item)
        if item is not None:
            accepted.set()
    server.queue.put=signal
    with ThreadPoolExecutor(1) as pool:
        future=pool.submit(server.client(),'last')
        assert accepted.wait(2)
        server.close()
        assert future.result()==[dict(frame='last',offset=(0,0),source='full')]
    assert not server.thread.is_alive()
