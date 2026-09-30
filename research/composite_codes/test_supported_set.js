"use strict";
const assert=require("node:assert/strict");
const s=require("./supported_set.js");
let checks=0;
function check(value,message){assert.ok(value,message);checks++;}
function near(a,b,tolerance=1e-9){check(Math.abs(a-b)<=tolerance,`${a} != ${b}`);}

near(s.chiSquareQuantile(0.95,1),3.841458820694124,1e-10);
near(s.chiSquareQuantile(0.95,2),5.991464547107979,1e-10);
near(s.chiSquareQuantile(0.95,6),12.591587243743977,1e-10);
near(s.chiSquareQuantile(0.95,0),0);
near(s.rayResidual([1,1],[1,0]),1);
near(s.rayResidual([-1,0],[1,0]),1);
near(s.coneResidual([1,1],[1,0],[0,1]),0);
near(s.coneResidual([-1,1],[1,0],[0,1]),1);
near(s.coneResidual([1,1],[1,0],[2,0]),1);

// A positive pair can coincide with a third pure ray although all pure rays differ.
const collision=s.prepareDesign([[1,0,0],[0,1,0],[0.5,0.5,0]],{nuisance:[],sigma:0.01});
const collisionResult=s.decide(collision,[0.5,0.5,0]);
check(collisionResult.supportedPure.includes(2),"Aliased pure retained");
check(collisionResult.supportedMixtures.includes(0),"True mixture retained");
check(collisionResult.certifiedPure===null,"Exact alias never certified pure");

// A pure signal separated from every eta-bounded pair is certifiable at high SNR.
const separated=s.prepareDesign([[1,0,0],[0,1,0]],{nuisance:[],sigma:0.01});
const pure=s.decide(separated,[1,0,0]);
check(pure.certifiedPure===0,"Separated pure certified");
check(s.decide(separated,[0,0,0]).certifiedPure===null,"Null not called pure");
check(s.decide(separated,[0,0,1]).spanRejected,"Out-of-span mean rejected at high SNR");

// Adding nuisance offsets cannot change evidence after nuisance projection.
const nuisance=s.prepareDesign([[1,-1,0,0],[0,0,1,-1]],{sigma:0.01});
const r1=s.decide(nuisance,[1,-1,0,0]),r2=s.decide(nuisance,[8,6,7,7]);
check(r1.certifiedPure===r2.certifiedPure,"Intercept invariance");
near(r1.pureResiduals[0],r2.pureResiduals[0]);

// A lossy measurement operator, e.g. a single integrated readout, destroys
// separation that is present in event space. Apply it BEFORE geometry.
const eventColumns=[[1,0],[0,1]];
const integratedColumns=eventColumns.map(v=>[v[0]+v[1]]);
const measured=s.prepareDesign(integratedColumns,{nuisance:[],sigma:0.01});
check(s.decide(measured,[1]).certifiedPure===null,"Measurement-induced alias honored");

// Joint changes of measurement and known-noise units leave all inference fixed.
for(const scale of [1e-12,1e-6,1,1e6,1e12]){
  const changed=s.prepareDesign([[scale,0,0],[0,scale,0]],{nuisance:[],sigma:0.01*scale});
  check(changed.rank===2,"Measurement-unit invariant numerical rank");
  check(s.decide(changed,[scale,0,0]).certifiedPure===0,"Measurement-unit invariant pure certificate");
  check(s.decide(changed,[0,0,scale]).spanRejected,"Measurement-unit invariant residual check");
}
check(s.prepareDesign([[1e-11,0,0],[0,1,0]],{nuisance:[],sigma:1e-12}).rank===2,"Small but independent columns not silently dropped");
assert.throws(()=>s.prepareDesign([[1,0,0],[1,1e-12,0]],{nuisance:[]}),/unresolved rank/);checks++;
assert.throws(()=>s.coneResidual([1,1],[1,0],[1,1e-7]),/unresolved composite cone/);checks++;

// A genuinely small mixture excluded by the scientific effect floor must NOT
// be advertised as protected: it can receive a pure certificate.
const belowFloor=s.decide(separated,[0.99,0.01,0]);
check(belowFloor.certifiedPure===0,"Below-floor limitation explicitly demonstrated");

const grid=s.transitionGrid();
check(grid.rows.length===240,"Correct grid size");
for(const c of grid.columns){near(c.reduce((a,b)=>a+b,0)/c.length,0);near(s.norm2(c)/c.length,1);}
near(grid.rows[0][3],grid.rows[0][0]/3);
check(s.LOCKED_COUNTS.reduce((a,b)=>a+b[1],0)===60,"Locked N60 counts");
const indices=s.LOCKED_COUNTS.flatMap(([id,n])=>Array(n).fill(id));
const actual=s.prepareDesign(grid.columns.map(c=>indices.map(i=>c[i])));
check(actual.rank===6,"Actual six-candidate geometry full rank after intercept");

// Validate the analytical Gaussian coverage mechanism independently of a
// particular classifier: every response with noise inside the simultaneous
// ball MUST retain its generating class, continuously over w in [eta,1-eta].
const normal=s.randomSource(871782911);let coveredChecks=0;
for(let rep=0;rep<1000;rep++){
  const p=actual.pairs[rep%actual.pairs.length],w=0.25+0.5*((rep*37)%997)/997;
  const mean=s.combination(actual.coordinates[p.i],actual.coordinates[p.j],w,1-w);
  const noise=actual.basis.map(()=>normal()),z=s.combination(mean,noise,1,1);
  const decision=s.decideCoordinates(actual,z,0);
  if(s.norm2(noise)<=actual.radius2){check(decision.supportedMixtures.includes(rep%actual.pairs.length),"Coverage implies true mixture retained");check(decision.certifiedPure===null,"Coverage implies no false purity");coveredChecks++;}
}
console.log(JSON.stringify({checks,coveredChecks,status:"passed"}));
