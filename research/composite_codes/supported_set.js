"use strict";

// Dependency-free reference implementation. Matrices are column arrays.
// Input covariance must already be whitened to sigma^2 I. Noise scale is KNOWN.
// prepareDesign/decide additionally divide by sigma; decideCoordinates expects
// unit-noise coordinates. Numerically unresolved rank/cones fail explicitly.
const CANDIDATE_NAMES = ["innovation_l2", "surprise", "gain", "update_l2", "information_gain", "change_probability"];
const LOCKED_COUNTS = [[0,3],[5,2],[11,1],[18,3],[23,3],[120,8],[125,8],[131,1],[138,8],[143,7],[192,3],[197,4],[210,4],[215,4],[219,1]];
function dot(a,b) { let s=0; for(let i=0;i<a.length;i++) s+=a[i]*b[i]; return s; }
function norm2(a) { return dot(a,a); }
function combination(a,b,s,t) { return a.map((v,i)=>s*v+t*b[i]); }
function projectOut(v,basis) { const r=v.slice(); for(let pass=0;pass<2;pass++) for(const q of basis) { const c=dot(r,q); for(let i=0;i<r.length;i++) r[i]-=c*q[i]; } return r; }
function orthonormalBasis(columns,tolerance=1e-10) {
  const basis=[];
  for(const c of columns) {
    const originalNorm=Math.sqrt(norm2(c));
    if(!Number.isFinite(originalNorm)||(originalNorm===0&&c.some(x=>x!==0)))throw new Error("Rescale input: floating-point norm overflow or underflow");
    const r=projectOut(c,basis),n=Math.sqrt(norm2(r));
    if(n>tolerance*originalNorm)basis.push(r.map(x=>x/n));
    else if(n>64*Number.EPSILON*originalNorm)throw new Error("Numerically unresolved rank: use a better-conditioned design or an independently justified rank reduction");
  }
  return basis;
}
function logGamma(x) {
  const c=[676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
  if(x<0.5) return Math.log(Math.PI)-Math.log(Math.sin(Math.PI*x))-logGamma(1-x);
  x-=1; let a=0.99999999999980993; for(let i=0;i<c.length;i++) a+=c[i]/(x+i+1);
  const t=x+c.length-0.5; return 0.5*Math.log(2*Math.PI)+(x+0.5)*Math.log(t)-t+Math.log(a);
}
function gammaP(a,x) {
  if(x<=0) return 0;
  if(x<a+1) { let term=1/a, sum=term; for(let n=1;n<10000;n++){term*=x/(a+n);sum+=term;if(Math.abs(term)<Math.abs(sum)*1e-15) break;} return sum*Math.exp(-x+a*Math.log(x)-logGamma(a)); }
  let b=x+1-a,c=1e300,d=1/b,h=d;
  for(let n=1;n<10000;n++){const an=-n*(n-a);b+=2;d=an*d+b;if(Math.abs(d)<1e-300)d=1e-300;c=b+an/c;if(Math.abs(c)<1e-300)c=1e-300;d=1/d;const delta=d*c;h*=delta;if(Math.abs(delta-1)<1e-15)break;}
  return 1-Math.exp(-x+a*Math.log(x)-logGamma(a))*h;
}
function chiSquareQuantile(probability,df) {
  if(!(probability>0&&probability<1)||!Number.isInteger(df)||df<0)throw new Error("Invalid chi-square arguments");
  if(df===0)return 0;
  let lo=0,hi=Math.max(1,df);while(gammaP(df/2,hi/2)<probability)hi*=2;
  for(let i=0;i<100;i++){const mid=(lo+hi)/2;if(gammaP(df/2,mid/2)<probability)lo=mid;else hi=mid;}return (lo+hi)/2;
}
function rayResidual(z,a) {
  const aa=norm2(a),b=aa>0?Math.max(0,dot(z,a)/aa):0;
  return norm2(combination(z,a,1,-b));
}
function coneResidual(z,a,b) {
  let best=Math.min(norm2(z),rayResidual(z,a),rayResidual(z,b));
  const aa=norm2(a),bb=norm2(b),ab=dot(a,b),az=dot(a,z),bz=dot(b,z),det=aa*bb-ab*ab;
  if(det>1e-12*aa*bb){const ca=(bb*az-ab*bz)/det,cb=(aa*bz-ab*az)/det;if(ca>=0&&cb>=0)best=Math.min(best,norm2(z.map((v,i)=>v-ca*a[i]-cb*b[i])));}
  else if(aa>0&&bb>0){
    const pivot=a.reduce((best,v,i)=>Math.abs(v)>Math.abs(a[best])?i:best,0);
    if(a.some((v,i)=>b[i]*a[pivot]!==v*b[pivot]))throw new Error("Numerically unresolved composite cone: nearly collinear, not exactly proportional");
  }
  return Math.max(0,best);
}
function prepareDesign(columns,{nuisance=null,sigma=1,alpha=0.05,beta=0.05,eta=0.25}={}) {
  if(!columns.length||!columns[0].length||!columns.every(x=>x.length===columns[0].length&&x.every(Number.isFinite)))throw new Error("Invalid design");
  if(!(sigma>0)||!Number.isFinite(sigma)||!(alpha>0&&alpha<1)||!(beta>0&&beta<1)||!(eta>0&&eta<=0.5))throw new Error("Invalid model settings");
  const n=columns[0].length;
  const nuisanceColumns=nuisance===null?[Array(n).fill(1)]:nuisance;
  if(!nuisanceColumns.every(x=>x.length===n&&x.every(Number.isFinite)))throw new Error("Invalid nuisance");
  const nuisanceBasis=orthonormalBasis(nuisanceColumns);
  const noiseScaledColumns=columns.map(x=>x.map(v=>v/sigma));
  if(!noiseScaledColumns.every(x=>x.every(Number.isFinite)))throw new Error("Rescale input: noise-unit design overflow");
  const residualColumns=noiseScaledColumns.map(x=>projectOut(x,nuisanceBasis));
  const basis=orthonormalBasis(residualColumns), coordinates=residualColumns.map(x=>basis.map(q=>dot(q,x)));
  const pairs=[];for(let i=0;i<columns.length;i++)for(let j=i+1;j<columns.length;j++)pairs.push({i,j,a:combination(coordinates[i],coordinates[j],1-eta,eta),b:combination(coordinates[i],coordinates[j],eta,1-eta)});
  const residualDf=n-nuisanceBasis.length-basis.length;
  return {n,sigma,alpha,beta,eta,nuisanceBasis,basis,coordinates,pairs,rank:basis.length,residualDf,coordinateNoiseStd:1,rankRelativeTolerance:1e-10,radius2:chiSquareQuantile(1-alpha,basis.length),residualThreshold:chiSquareQuantile(1-beta,residualDf)};
}
function decideCoordinates(design,z,residualSquared=0) {
  if(z.length!==design.rank||!z.every(Number.isFinite)||!Number.isFinite(residualSquared)||residualSquared<0)throw new Error("Invalid observation coordinates");
  const pureResiduals=design.coordinates.map(x=>rayResidual(z,x));
  const mixtureResiduals=design.pairs.map(p=>coneResidual(z,p.a,p.b));
  const confidenceThreshold=design.radius2*(1+1e-10),adequacyThreshold=design.residualThreshold*(1+1e-10);
  const supportedPure=pureResiduals.map((v,i)=>v<=confidenceThreshold?i:-1).filter(i=>i>=0);
  const supportedMixtures=mixtureResiduals.map((v,i)=>v<=confidenceThreshold?i:-1).filter(i=>i>=0);
  const nullSupported=norm2(z)<=confidenceThreshold;
  const spanRejected=residualSquared>adequacyThreshold;
  const noneCompatible=!nullSupported&&!supportedPure.length&&!supportedMixtures.length;
  const certifiedPure=!spanRejected&&!nullSupported&&supportedPure.length===1&&!supportedMixtures.length?supportedPure[0]:null;
  return {supportedPure,supportedMixtures,nullSupported,spanRejected,noneCompatible,certifiedPure,pureResiduals,mixtureResiduals};
}
function decide(design,y) {
  if(y.length!==design.n||!y.every(Number.isFinite))throw new Error("Invalid response");
  const scaled=y.map(v=>v/design.sigma);
  if(!scaled.every(Number.isFinite))throw new Error("Rescale input: noise-unit observation overflow");
  const residual=projectOut(scaled,design.nuisanceBasis),z=design.basis.map(q=>dot(q,residual));
  return decideCoordinates(design,z,Math.max(0,norm2(residual)-norm2(z)));
}
function digamma(x){let r=0;while(x<12){r-=1/x;x+=1;}const q=1/(x*x);return r+Math.log(x)-0.5/x-q*(1/12-q*(1/120-q*(1/252-q*(1/240-q/132))));}
function transitionGrid(){
  const rows=[];
  for(const q of [0.05,0.15,0.35,0.65,0.90])for(const shape of [0.5,0.9])for(const c of [2,8,32,128])for(const reset of [0.05,0.5,0.95])for(const hazard of [0.01,0.15]){
    const innovation=(1-q)*Math.sqrt(1+shape*shape+(1-shape)*(1-shape));
    rows.push([innovation,-Math.log(q),1/(c+1),innovation/(c+1),-Math.log(q)+digamma(c*q+1)-digamma(c+1),hazard*reset/((1-hazard)*q+hazard*reset)]);
  }
  const rawColumns=CANDIDATE_NAMES.map((_,j)=>rows.map(r=>r[j]));
  const columns=rawColumns.map(c=>{const m=c.reduce((a,b)=>a+b,0)/c.length,s=Math.sqrt(c.reduce((a,b)=>a+(b-m)**2,0)/c.length);return c.map(x=>(x-m)/s);});
  return {rows,rawColumns,columns};
}
function randomSource(seed){let s=seed>>>0;if(!s)s=1;let spare=null;const uniform=()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return ((s>>>0)+0.5)/4294967296;};return ()=>{if(spare!==null){const v=spare;spare=null;return v;}const r=Math.sqrt(-2*Math.log(uniform())),theta=2*Math.PI*uniform();spare=r*Math.sin(theta);return r*Math.cos(theta);};}
function seededKey(seed,generator,replicate){let x=(seed^Math.imul(generator+1,0x9e3779b1)^Math.imul(replicate+1,0x85ebca6b))>>>0;x^=x>>>16;x=Math.imul(x,0x7feb352d);x^=x>>>15;x=Math.imul(x,0x846ca68b);x^=x>>>16;return x>>>0;}
function wilson(successes,total){const p=successes/total,z=1.959963984540054,z2=z*z,d=1+z2/total,c=(p+z2/(2*total))/d,r=z*Math.sqrt(p*(1-p)/total+z2/(4*total*total))/d;return [Math.max(0,c-r),Math.min(1,c+r)];}

module.exports={CANDIDATE_NAMES,LOCKED_COUNTS,dot,norm2,combination,projectOut,orthonormalBasis,logGamma,gammaP,chiSquareQuantile,rayResidual,coneResidual,prepareDesign,decideCoordinates,decide,digamma,transitionGrid,randomSource,seededKey,wilson};
