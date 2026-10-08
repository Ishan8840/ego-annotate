"""Request isolation, failure propagation, and bounded cache regression checks."""
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import io,json,queue,threading,types
import pytest
from egoannot.qwen_runtime import QwenRuntime


def fake_runtime(fail=False):
    r=object.__new__(QwenRuntime)
    r.closed=False;r.state_lock=threading.Lock();r.max_batch_size=4;r.batch_wait_s=.05
    r.queue=queue.Queue();r.local=threading.local();r.stats=[];r.trace_dir=None
    r.torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:None))
    def infer(requests):
        if fail:raise ValueError('inference failed')
        return [(json.dumps({'span_id':q.batch[0]['span_id'],'text':q.system}),
                 {'input_tokens':1,'output_tokens':1}) for q in requests]
    r._infer=infer
    r.worker=threading.Thread(target=r._work,daemon=True);r.worker.start()
    return r


def test_concurrent_replies_and_last_raw_stay_with_request():
    r=fake_runtime()
    try:
        def run(i):
            got,_=r(str(i),[],[{'span_id':str(i)}])
            return got,json.loads(r.last_raw)
        with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,range(12)))
        for i,(objects,raw) in enumerate(results):
            assert objects==[{'span_id':str(i),'text':str(i)}]
            assert raw==objects[0]
        assert any(x['batch_size']>1 for x in r.stats)
    finally:r.close()


def test_batch_failure_reaches_every_waiter_and_close_is_idempotent():
    r=fake_runtime(fail=True)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending=[pool.submit(r,'test',[],[{'span_id':str(i)}]) for i in range(4)]
            for future in pending:
                with pytest.raises(ValueError,match='inference failed'):future.result(timeout=2)
    finally:r.close();r.close()
    with pytest.raises(RuntimeError,match='closed'):r('test',[],[])


def test_decoded_image_cache_uses_content_and_evicts_without_changing_pixels():
    from PIL import Image
    r=object.__new__(QwenRuntime);r.images=OrderedDict();r.cache_bytes=0;r.cache_limit=2*8*8*3
    def blob(color):
        stream=io.BytesIO();Image.new('RGB',(8,8),color).save(stream,'PNG');return stream.getvalue()
    red,blue,green=[blob(c) for c in ['red','blue','green']]
    first=r._image(red);assert r._image(red) is first
    r._image(blue);r._image(green)
    assert len(r.images)==2 and r.cache_bytes==r.cache_limit
    assert r._image(red).tobytes()==first.tobytes()


def test_incomplete_batch_fails_instead_of_leaving_callers_waiting():
    r=fake_runtime()
    r._infer=lambda requests: []
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending=pool.submit(r,'test',[],[])
            with pytest.raises(RuntimeError,match='Incomplete inference batch'):
                pending.result(timeout=2)
    finally:
        r.close()
