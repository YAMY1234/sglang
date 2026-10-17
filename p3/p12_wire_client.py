"""Exercise one real P/D pair for P12 wire proofs; never a timing workload."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import secrets
import urllib.error
import urllib.request


def post(url, body, timeout):
    request=urllib.request.Request(url+'/generate',data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=timeout) as response:
            return dict(status=response.status,body=json.loads(response.read()))
    except urllib.error.HTTPError as error:
        raw=error.read().decode(errors='replace')
        try:raw=json.loads(raw)
        except ValueError:pass
        return dict(status=error.code,body=raw)
    except (urllib.error.URLError,TimeoutError) as error:
        return dict(status=0,error=str(error))


def successful(response):
    body=response.get('body')
    if response['status']!=200 or not isinstance(body,dict):return False
    meta=body.get('meta_info',{});reason=meta.get('finish_reason') or {}
    return (not body.get('error') and reason.get('type')!='abort'
        and meta.get('completion_tokens',0)>=2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p',required=True);parser.add_argument('--d',required=True)
    parser.add_argument('--bootstrap-host',required=True)
    parser.add_argument('--bootstrap-port',required=True,type=int)
    parser.add_argument('--mode',choices=('positive','event','abort'),default='positive')
    parser.add_argument('--timeout',type=int,default=600)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();nonce=secrets.token_hex(6)
    cases=[(1025,False),(4097,False)] if args.mode=='positive' else [(1025,True)]+[(1025,False)]*5
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,(length,negative) in enumerate(cases):
            prefix=f'p12-negative-{args.mode}' if negative else 'pdtune-isolated-p12'
            rid=f'{prefix}-{nonce}-{i}'
            body=dict(rid=rid,input_ids=[1148]*length,stream=False,
                sampling_params=dict(max_new_tokens=2,temperature=0,ignore_eos=True),
                bootstrap_host=args.bootstrap_host,bootstrap_port=args.bootstrap_port,
                bootstrap_room=secrets.randbits(61))
            futures={role:pool.submit(post,url.rstrip('/'),body,args.timeout)
                for role,url in (('P',args.p),('D',args.d))}
            responses={role:future.result() for role,future in futures.items()}
            # Negative failure must be explicit. A dead connection/timeout is
            # not evidence that the original abort/drain protocol worked.
            d=responses['D'];finish=(d.get('body') or {}) if isinstance(d.get('body'),dict) else {}
            aborted=(finish.get('meta_info',{}).get('finish_reason') or {}).get('type')=='abort'
            passed=(d['status']>=400 or aborted) if negative else successful(d)
            results.append(dict(rid=rid,length=length,negative=negative,passed=passed,responses=responses))
            args.out.write_text(json.dumps(dict(passed=all(x['passed'] for x in results),
                complete=len(results)==len(cases),timing_eligible=False,mode=args.mode,requests=results),indent=2)+'\n')
            if not passed:break
    if len(results)!=len(cases) or not all(x['passed'] for x in results):raise SystemExit(1)
    print('P12_WIRE_CLIENT_PASS',args.mode,len(results),flush=True)


if __name__=='__main__':main()
