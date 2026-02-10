export class Idefics3PreTrainedModel extends PreTrainedModel {
}
/**
 * The Idefics3 model which consists of a vision backbone and a language model.
 */
export class Idefics3ForConditionalGeneration extends Idefics3PreTrainedModel {
    encode_image({ pixel_values, pixel_attention_mask }: {
        pixel_values: any;
        pixel_attention_mask: any;
    }): Promise<any>;
    _merge_input_ids_with_image_features(kwargs: any): {
        inputs_embeds: any;
        attention_mask: any;
    };
}
/**
 * The SmolVLM Model with a language modeling head.
 * It is made up a SigLIP vision encoder, with a language modeling head on top.
 */
export class SmolVLMForConditionalGeneration extends Idefics3ForConditionalGeneration {
}
import { PreTrainedModel } from '../modeling_utils.js';
//# sourceMappingURL=modeling_idefics3.d.ts.map