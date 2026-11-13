// OpenCode plugin: registers this repo's skills/ directory as a skill source,
// so OpenCode discovers every skill in it without copying files.
import { fileURLToPath } from "node:url";

const skillsDir = fileURLToPath(new URL("../skills", import.meta.url));

export default {
  id: "synalinks",
  server: async () => ({
    config: async (config) => {
      config.skills = config.skills ?? {};
      config.skills.paths = [...(config.skills.paths ?? []), skillsDir];
    },
  }),
};
