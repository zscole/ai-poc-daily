export class PaliGemmaPreTrainedModel extends PreTrainedModel {
}
export class PaliGemmaForConditionalGeneration extends PaliGemmaPreTrainedModel {
    _merge_input_ids_with_image_features(kwargs: any): {
        inputs_embeds: any;
        attention_mask: any;
    };
}
import { PreTrainedModel } from '../modeling_utils.js';
//# sourceMappingURL=modeling_paligemma.d.ts.map