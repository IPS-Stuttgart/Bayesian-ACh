"use strict";
// Run from the local package. Output paths are explicit and never overwrite.
const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto");
const s=require("./supported_set.js");
const CONFIG={seed:20260930,replicates:2000,alpha:0.05,beta:0.05,eta:0.25,sigma:1,repetitionFactors:[1,4,16],weights:[0.25,0.5,0.75]};
function sd(x){const m=x.reduce((a,b)=>a+b,0)/x.length;return Math.sqrt(x.reduce((a,b)=>a+(b-m)**2,0)/x.length);}
function unitSd(x){const scale=sd(x);return x.map(v=>v/scale);}
function evaluate(){
  const grid=s.transitionGrid(),baseIndices=s.LOCKED_COUNTS.flatMap(([id,n])=>Array(n).fill(id));
  const generators=grid.columns.map((v,j)=>({kind:"pure",name:s.CANDIDATE_NAMES[j],pure:j,values:v}));
  let pair=0;for(let i=0;i<6;i++)for(let j=i+1;j<6;j++,pair++)for(const w of CONFIG.weights)generators.push({kind:"mixture",name:s.CANDIDATE_NAMES[i]+"+"+s.CANDIDATE_NAMES[j],pair,weight:w,values:unitSd(s.combination(grid.columns[i],grid.columns[j],w,1-w))});
  generators.push({kind:"null",name:"null",values:Array(240).fill(0)});
  const fullBasis=s.orthonormalBasis([Array(240).fill(1),...grid.columns]);
  generators.push({kind:"nonlinear_probe",name:"full_grid_orthogonalized_tanh_surprise",values:unitSd(s.projectOut(grid.columns[1].map(Math.tanh),fullBasis))});
  const rows=[];
  for(const repetition of CONFIG.repetitionFactors){
    const indices=Array.from({length:repetition},()=>baseIndices).flat();
    const d=s.prepareDesign(grid.columns.map(c=>indices.map(i=>c[i])),CONFIG);
    for(let g=0;g<generators.length;g++){
      const generator=generators[g],responseMean=indices.map(i=>generator.values[i]);
      const residualMean=s.projectOut(responseMean,d.nuisanceBasis),mean=d.basis.map(q=>s.dot(q,residualMean));
      const meanOut=Math.sqrt(Math.max(0,s.norm2(residualMean)-s.norm2(mean)));
      let correctPure=0,falsePure=0,anyPure=0,trueSupported=0,nullSupported=0,noneCompatible=0,spanRejected=0,supportSizeSum=0;
      for(let rep=0;rep<CONFIG.replicates;rep++){
        const normal=s.randomSource(s.seededKey(CONFIG.seed+repetition,g,rep));
        const z=mean.map(v=>v+CONFIG.sigma*normal());
        let residualSquared=0;
        for(let k=0;k<d.residualDf;k++){const noise=CONFIG.sigma*normal()+(k===0?meanOut:0);residualSquared+=noise*noise;}
        const decision=s.decideCoordinates(d,z,residualSquared);
        anyPure+=Number(decision.certifiedPure!==null);
        correctPure+=Number(generator.kind==="pure"&&decision.certifiedPure===generator.pure);
        falsePure+=Number(decision.certifiedPure!==null&&(generator.kind!=="pure"||decision.certifiedPure!==generator.pure));
        if(generator.kind==="pure")trueSupported+=Number(decision.supportedPure.includes(generator.pure));
        else if(generator.kind==="mixture")trueSupported+=Number(decision.supportedMixtures.includes(generator.pair));
        else if(generator.kind==="null")trueSupported+=Number(decision.nullSupported);
        nullSupported+=Number(decision.nullSupported);noneCompatible+=Number(decision.noneCompatible);spanRejected+=Number(decision.spanRejected);
        supportSizeSum+=decision.supportedPure.length+decision.supportedMixtures.length+Number(decision.nullSupported);
      }
      rows.push({budget:indices.length,kind:generator.kind,generator:generator.name,weight:generator.weight??null,replicates:CONFIG.replicates,correctPure,falsePure,anyPure,trueSupported,correctPureRate:correctPure/CONFIG.replicates,falsePureRate:falsePure/CONFIG.replicates,falsePureWilson:s.wilson(falsePure,CONFIG.replicates),trueSupportRate:generator.kind==="nonlinear_probe"?null:trueSupported/CONFIG.replicates,nullSupportRate:nullSupported/CONFIG.replicates,noneCompatibleRate:noneCompatible/CONFIG.replicates,spanRejectedRate:spanRejected/CONFIG.replicates,meanSupportSize:supportSizeSum/CONFIG.replicates,rank:d.rank});
    }
  }
  const summary=CONFIG.repetitionFactors.map(f=>{const selected=rows.filter(x=>x.budget===60*f),pure=selected.filter(x=>x.kind==="pure"),mixtures=selected.filter(x=>x.kind==="mixture");return {budget:60*f,minimumPureCertification:Math.min(...pure.map(x=>x.correctPureRate)),maximumMixtureFalsePure:Math.max(...mixtures.map(x=>x.falsePureRate)),minimumMixtureTrueSupport:Math.min(...mixtures.map(x=>x.trueSupportRate)),maximumMixtureMeanSupportSize:Math.max(...mixtures.map(x=>x.meanSupportSize)),maximumMixtureFalsePureWilsonUpper:Math.max(...mixtures.map(x=>x.falsePureWilson[1])),nullFalsePure:selected.find(x=>x.kind==="null").falsePureRate,nonlinearProbeFalsePure:selected.find(x=>x.kind==="nonlinear_probe").falsePureRate};});
  return {schemaVersion:1,experiment:"known_covariance_continuous_bounded_mixture_supported_set",config:CONFIG,scope:"Analytic false-pure bound applies only to declared nonnegative pure and eta-bounded two-candidate means under known Gaussian covariance and fixed design; no arbitrary open-set or biological guarantee.",summary,rows};
}
function main(){const out=process.argv[2];if(!out)throw new Error("Usage: node run_supported_set.js OUTPUT_DIRECTORY");if(fs.existsSync(out))throw new Error("Output exists; refusing to overwrite");const result=evaluate();fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,"result.json"),JSON.stringify(result,null,2)+"\n");const names=["protocol.md","supported_set.js","test_supported_set.js","independent_verify.py","run_supported_set.js"],manifest={kind:"post_failure_supported_set_diagnostic",sources:names.map(name=>({name,sha256:crypto.createHash("sha256").update(fs.readFileSync(path.join(__dirname,name))).digest("hex")})),payload:{name:"result.json",sha256:crypto.createHash("sha256").update(fs.readFileSync(path.join(out,"result.json"))).digest("hex")}};fs.writeFileSync(path.join(out,"manifest.json"),JSON.stringify(manifest,null,2)+"\n");console.log(JSON.stringify({summary:result.summary,manifest},null,2));}
if(require.main===module)main();
module.exports={CONFIG,evaluate};
