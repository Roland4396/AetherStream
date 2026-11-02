/* Real Chromium + iPhone WebKit controls against the isolated FastAPI fixture. */
const {chromium, webkit, devices} = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
process.umask(0o077);
const directory = process.argv[2];
if (!directory) throw new Error('Pass the private fixture directory');
const fixture = JSON.parse(fs.readFileSync(path.join(directory, 'fixture.json'), 'utf8'));
const shots = path.join(directory, 'screenshots');fs.mkdirSync(shots, {recursive: true});
const records = [];
const inventory = {
  requirements: ['default follows caller', 'one channel selector', 'exact model overrides only its channel', 'visible effective mode and source', 'revision conflict prevents stale saves', 'no inference'],
  controls: ['login including invalid/empty token', 'filter and select channel', 'channel mode/save', 'model preview', 'model override/save/remove', 'refresh', 'logout'],
  offHappyPath: ['stale second-page save', 'catalog text that looks like HTML is never executed', 'new channel defaults to follow', 'token is not persisted'],
  views: ['desktop loaded', 'desktop model override', 'iPhone WebKit loaded', 'iPhone model override', 'conflict banner'],
};
fs.writeFileSync(path.join(directory, 'qa-inventory.json'), JSON.stringify(inventory, null, 2));

async function api(method, payload) {
  const response = await fetch(fixture.url + '/admin/request-modes/state', {
    method, headers: {Authorization: 'Bearer ' + fixture.token, ...(payload?{'Content-Type':'application/json'}:{})},
    ...(payload?{body:JSON.stringify(payload)}:{}),
  });
  assert.equal(response.status,200);return response.json();
}
async function reset() {
  let state=await api('GET');
  state=await api('POST',{revision:state.revision,channel:'anti',model:null,mode:'auto'});
  for(const model of state.channels.find(c=>c.id==='anti').models){state=await api('POST',{revision:state.revision,channel:'anti',model:model.upstream_model,mode:'inherit'});}
}
async function message(page, pattern) {
  await page.waitForFunction(({source, flags}) => new RegExp(source, flags).test(document.getElementById('message').textContent), {source:pattern.source,flags:pattern.flags});
}
async function login(page) {
  await page.goto(fixture.url+'/admin/request-modes');
  await page.locator('#token').fill(fixture.token);
  await page.locator('#connect').click();
  await page.locator('#console').waitFor({state:'visible'});
}
async function fit(page) {
  const size=await page.evaluate(()=>({width:innerWidth, client:document.documentElement.clientWidth, scroll:document.documentElement.scrollWidth}));
  assert.ok(size.scroll<=size.client,JSON.stringify(size));return size;
}

(async()=>{
 for(const [name,type,options] of [
   ['chromium',chromium,{viewport:{width:1280,height:960}}],
   ['iphone-webkit',webkit,{...devices['iPhone 13']}],
 ]){
  await reset();const browser=await type.launch({headless:true});const context=await browser.newContext(options);const page=await context.newPage();const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  try{
   await page.goto(fixture.url+'/admin/request-modes');await page.locator('#connect').click();await message(page,/请输入私有管理令牌/);
   await page.locator('#token').fill('invalid');await page.locator('#connect').click();await message(page,/Unauthorized/);
   await page.locator('#token').fill(fixture.token);await page.locator('#token').press('Enter');await page.locator('#console').waitFor({state:'visible'});
   assert.equal(await page.locator('#channel-mode').inputValue(),'auto');assert.match(await page.locator('#channel-effect').textContent(),/选非流就发非流/);
   await page.screenshot({path:path.join(shots,name+'-initial.png'),scale:'css'});await fit(page);
   await page.getByRole('button',{name:'subscription',exact:true}).click();assert.equal(await page.locator('#channel-mode').inputValue(),'stream');
   await page.getByRole('button',{name:'fresh-channel',exact:true}).click();assert.equal(await page.locator('#channel-mode').inputValue(),'auto');
   await page.getByRole('button',{name:'anti',exact:true}).click();await page.locator('#channel-filter').fill('anti');assert.equal(await page.locator('#channels button').count(),1);await page.locator('#channel-filter').fill('');
   const other=await context.newPage();await login(other);
   await page.locator('#channel-mode').selectOption('json');await page.locator('#save-channel').click();await message(page,/已保存/);
   assert.match(await page.locator('#channel-effective').textContent(),/固定非流式.*渠道设置/);
   await other.locator('#channel-mode').selectOption('stream');await other.locator('#save-channel').click();await message(other,/其他页面修改/);
   assert.equal((await api('GET')).channels.find(c=>c.id==='anti').mode,'json');
   await other.screenshot({path:path.join(shots,name+'-conflict.png'),scale:'css'});await other.close();
   await page.locator('#public-model').fill('[按量]gemini-4-argon');await page.locator('#check-model').click();await page.locator('#preview').waitFor({state:'visible'});
   assert.match(await page.locator('#preview').textContent(),/gemini-4-argon.*固定非流式/s);
   await page.locator('#preview select').selectOption('auto');await page.locator('#preview button').click();await message(page,/已保存/);
   assert.equal(await page.locator('.model-row').count(),1);assert.match(await page.locator('.model-row').textContent(),/跟随酒馆.*模型单独设置/s);assert.equal(await page.locator('#channel-mode').inputValue(),'json');
   await page.locator('#model-list').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(shots,name+'-model-override.png'),scale:'css'});await fit(page);
   await page.locator('.model-row select').selectOption('inherit');await page.locator('.model-row button').click();await message(page,/已保存/);assert.equal(await page.locator('.model-row').count(),0);
   await page.locator('#public-model').fill('<img src=x onerror=window.PWNED=1>');await page.locator('#check-model').click();await page.locator('#preview').waitFor({state:'visible'});
   assert.equal(await page.locator('#preview img').count(),0);assert.equal(await page.evaluate(()=>window.PWNED),undefined);
   await page.locator('#refresh').click();await message(page,/已读取/);
   await page.locator('#channel-mode').selectOption('auto');await page.locator('#save-channel').click();await message(page,/已保存/);
   assert.equal(await page.evaluate(()=>localStorage.length+sessionStorage.length),0);
   await page.locator('#disconnect').click();await page.locator('#login').waitFor({state:'visible'});assert.equal(await page.locator('#token').inputValue(),'');
   assert.equal(JSON.parse(fs.readFileSync(fixture.config)).untouched.test_marker,'must survive every save');
   assert.deepEqual(errors,[]);records.push({browser:name,checks:'all controls and off-happy paths passed',fit:await fit(page),noPageErrors:true});
  }finally{await context.close();await browser.close();}
 }
 fs.writeFileSync(path.join(directory,'ui-results.json'),JSON.stringify(records,null,2));console.log(JSON.stringify(records));
})().catch(error=>{console.error(error);process.exitCode=1;});
