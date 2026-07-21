interface DriftResponse{
    drift_detected : boolean,
    drift_share : number,
    drifted_features : string[],
    n_drifted : number,
    n_total : number
}

export async function handler() {
    const modelUrl = process.env.MODEL_SERVICE_URL;
    if (!modelUrl) {
        console.error("MODEL_SERVICE_URL not set");
        return { drift_checked: false, reason: "MODEL_SERVICE_URL not set" };
    }

    const driftRes = await fetch(`${modelUrl}/drift`, {
        method: "POST",
        headers: { "x-model-secret": process.env.MODEL_SECRET! },
    });
    if (!driftRes.ok) {
        console.error("Drift endpoint failed:", driftRes.status, await driftRes.text());
        return { drift_checked: false, reason: `drift endpoint returned ${driftRes.status}` };
    }

    const drift: DriftResponse = await driftRes.json();

    if (drift.drift_detected) {
        const token = process.env.GH_TOKEN;
        if (!token) {
            console.error("No GH token set");
            return { drift_checked: true, dispatch_triggered: false, reason: "no GH token" };
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
            },
        );

        if (!ghRes.ok) {
            console.error("Failed to dispatch retrain", await ghRes.text());
            return { drift_checked: true, dispatch_triggered: false, reason: "API error" };
        }
    }

    return { drift_checked: true, dispatch_triggered: drift.drift_detected };
}
