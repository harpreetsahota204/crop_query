import { registerComponent, PluginComponentType } from "@fiftyone/plugins";
import CropQueryPanel from "./CropQueryPanel";

console.log("[CropQuery] JS bundle loaded — registering component");

registerComponent({
  name: "CropQueryPanel",
  component: CropQueryPanel,
  type: PluginComponentType.Component,
});

console.log("[CropQuery] Component registered successfully");
