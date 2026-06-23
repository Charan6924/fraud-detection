interface DriftResponse{
    drift_detected : boolean,
    drift_share : number,
    drifted_features : string[],
    n_drifted : number,
    n_total : number
}

export async function handler(){
    const modelUrl = process.env.MODEL_SERVICE_URL;
    if (!modelUrl){
        return;
    }

    const driftRes = await fetch(`${modelUrl}/drift`, { method: "POST" });   
    if (!driftRes.ok) {                                                                                                                                    
      console.error("Drift endpoint failed:", driftRes.status, await driftRes.text());                                                                     
      return;                                                                                                                                              
    }     

    const drift : DriftResponse = await driftRes.json()

    if (drift.drift_detected){
        // call github action and retrain model
        const token = process.env.GH_TOKEN;
        if (!token){
            console.error("No GH token set")
            return
        }

        const ghRes = await fetch(                                                                                                                           
        "https://api.github.com/repos/Charan6924/fraud-detection/dispatches",                                                                              
        {                                                                                                                                                  
          method: "POST",                                                                                                                                  
          headers: {                                                                                                                                       
            Accept: "application/vnd.github+json",                                                                                                         
            Authorization: `Bearer ${token}`,                                                                                                              
            "User-Agent": "fraud-detection-monitor",                                                                                                       
          },                                                                                                                                               
          body: JSON.stringify({ event_type: "retrain" }),                                                                                                 
        },);
        
        if (!ghRes.ok) {
            console.error("Failed to dispatch retrain", await ghRes.text());
        }

    }
}
