// Deterministic canvas frames -> H.264 MP4. PLAYWRIGHT_PATH can point to an existing install.
const {chromium}=require(process.env.PLAYWRIGHT_PATH || require.resolve('playwright',{paths:[__dirname,'/tmp/ego-review-browser']}));
const {spawn}=require('child_process');const fs=require('fs');const path=require('path');const {once}=require('events');
const root=path.resolve(__dirname,'..'),out=path.join(root,'artifacts/demo');
const preview=process.argv.includes('--stills');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/opt/google/chrome/chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage','--allow-file-access-from-files']});
 const page=await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 if(process.argv.includes('--verify-delivery')){
  await page.goto('http://127.0.0.1:18081/demo/',{waitUntil:'networkidle'});
  await page.waitForFunction(()=>document.querySelector('video').readyState>=2);
  if(/preload/i.test(await page.locator('body').innerText()))throw new Error('Unexpected branding on delivery page.');
  const video=page.locator('video');
  const state=await video.evaluate(v=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration}));
  if(state.width!==1920||state.height!==1080||Math.abs(state.duration-30)>.05)throw new Error(JSON.stringify(state));
  const seeks=[];
  for(const time of [3.25,7,12.5,17,20,22,24,27,29]){
   await video.evaluate((v,t)=>{v.currentTime=t},time);
   await page.waitForFunction(t=>{const v=document.querySelector('video');return !v.seeking&&Math.abs(v.currentTime-t)<.15;},time);
   seeks.push(await video.evaluate(v=>v.currentTime));
  }
  await video.evaluate(v=>{v.muted=true;return v.play()});await page.waitForTimeout(500);await video.evaluate(v=>v.pause());
  const mediaError=await video.evaluate(v=>v.error?.message||null);if(mediaError||errors.length)throw new Error(JSON.stringify({mediaError,errors}));
  const qualityResponse=await page.request.get('http://127.0.0.1:18081/demo/quality/analysis.json');
  if(!qualityResponse.ok())throw new Error('Quality evidence download failed.');
  const quality=(await qualityResponse.json()).summary;
  if(quality.clips!==2||quality.total_frames!==1440)throw new Error('Quality report does not match demo sources.');
  if(await page.locator('a[href="quality/"]').count()!==1)throw new Error('Missing quality report link.');
  const scoreResponse=await page.request.get('http://127.0.0.1:18081/demo/quality/composite.json');
  if(!scoreResponse.ok())throw new Error('Combined score evidence download failed.');
  const score=await scoreResponse.json();
  const localScore=JSON.parse(fs.readFileSync(path.join(root,'artifacts/color-demo/quality/composite.json'),'utf8'));
  if(score.score_percent!==localScore.score_percent || Object.keys(score.components).length!==8)throw new Error('Combined score mismatch.');
  if(await page.locator('a[href="quality/composite.html"]').count()!==1)throw new Error('Missing score breakdown link.');
  const result={...state,seeks_s:seeks,playback:true,mediaError,javascript_errors:errors,
   quality_evidence:{clips:quality.clips,frames:quality.total_frames,estimated_overall_quality_percent:score.score_percent}};
  fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
  await browser.close();return;
 }
 await page.goto('file://'+path.join(__dirname,'preload-film.html')+'?render=1');await page.evaluate(()=>window.ready);
 if(!preview && await page.evaluate(()=>DATA.preview_only))throw new Error('Cannot export incomplete preview annotations.');
 if(preview){for(const t of [1.5,6.5,12.5,17,20,22.5,24,27,29]){await page.evaluate(t=>window.render(t),t);await page.screenshot({path:path.join(out,`still-${t}.png`)});}console.log('Rendered design frames.');}
 else {
  const dest=path.join(out,'ego-demo-silent.mp4');
  const ff=spawn('ffmpeg',['-y','-v','warning','-f','image2pipe','-vcodec','mjpeg','-framerate','30','-i','-',
   '-an','-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p',
   '-c:v','libx264','-preset','slow','-crf','18','-maxrate','12M','-bufsize','24M','-pix_fmt','yuv420p',
   '-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-movflags','+faststart',dest],{stdio:['pipe','ignore','pipe']});
  let log='';ff.stderr.on('data',d=>log+=d);const done=once(ff,'close');
  for(let i=0;i<900;i++){await page.evaluate(t=>window.render(t),i/30);const frame=await page.screenshot({type:'jpeg',quality:95});if(!ff.stdin.write(frame))await once(ff.stdin,'drain');if(i%150===0)console.log(`Rendered ${i}/900 frames`);}
  ff.stdin.end();const [code]=await done;if(code!==0)throw new Error(log);console.log(dest);
 }
 if(errors.length)throw new Error(errors.join('\n'));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
