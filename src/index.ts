import { Container } from "@cloudflare/containers";

// The container running the FastAPI app (see Dockerfile)
export class FruitApp extends Container {
  defaultPort = 7860;      // uvicorn listens here (Dockerfile CMD)
  sleepAfter = "10m";      // go to sleep after 10 minutes with no visitors → no cost while idle
  envVars = {
    MODEL_PATH: "adam_lr1e-3.pt",
    THRESHOLD: "0.80",     // below this confidence → "ส่งให้พนักงานตรวจ"
  };
}

interface Env {
  FRUIT_APP: DurableObjectNamespace<FruitApp>;
}

// Every request goes to one shared container instance
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return env.FRUIT_APP.getByName("main").fetch(request);
  },
};
