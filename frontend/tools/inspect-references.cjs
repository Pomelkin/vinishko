const { chromium } = require('@playwright/test');
const path = require('path');
const fs = require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 const page=await browser.newPage({viewport:{width:390,height:844},deviceScaleFactor:1,javaScriptEnabled:false});
 await page.route('https://**',r=>r.abort());
 for (const name of ['Scaner','Download','NotFound','Vino','Similar']){
  await page.goto('file:///'+path.resolve(__dirname,'../..',name+'.mhtml').replace(/\\/g,'/'));
  await page.screenshot({path:path.resolve(__dirname,'../references',name+'.png'),fullPage:false});
  const info=await page.evaluate(()=>({text:document.body.innerText.slice(0,9000), images:[...document.images].slice(0,35).map(i=>({src:i.getAttribute('src'),alt:i.alt,width:i.width,height:i.height})), styles:[...document.querySelectorAll('h1,h2,button')].slice(0,20).map(e=>({text:e.textContent.slice(0,100),font:getComputedStyle(e).font,color:getComputedStyle(e).color,bg:getComputedStyle(e).backgroundColor,radius:getComputedStyle(e).borderRadius}))}));
  fs.writeFileSync(path.resolve(__dirname,'../references',name+'.json'),JSON.stringify(info,null,2));
 }
 await browser.close();
})();
