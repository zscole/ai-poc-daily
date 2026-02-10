export class Qwen2VLPreTrainedModel extends PreTrainedModel {
}
export class Qwen2VLForConditionalGeneration extends Qwen2VLPreTrainedModel {
    /**
     * Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
     *
     * Explanation:
     *     Each embedding sequence contains vision embedding and text embedding or just contains text embedding.
     *
     *     For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
     *     Examples:
     *         input_ids: [T T T T T], here T is for text.
     *         temporal position_ids: [0, 1, 2, 3, 4]
     *         height position_ids: [0, 1, 2, 3, 4]
     *         width position_ids: [0, 1, 2, 3, 4]
     *
     *     For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
     *     and 1D rotary position embeddin for text part.
     *     Examples:
     *         Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
     *         input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
     *         vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
     *         vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
     *         vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
     *         text temporal position_ids: [3, 4, 5, 6, 7]
     *         text height position_ids: [3, 4, 5, 6, 7]
     *         text width position_ids: [3, 4, 5, 6, 7]
     *         Here we calculate the text start position_ids as the max vision position_ids plus 1.
     *
     * @param {Tensor} input_ids Indices of input sequence tokens in the vocabulary. Tensor of shape `(batch_size, sequence_length)`.
     * @param {Tensor} image_grid_thw (Optional) The temporal, height and width of feature shape of each image in LLM. Tensor of shape `(num_images, 3)`.
     * @param {Tensor} video_grid_thw (Optional) The temporal, height and width of feature shape of each video in LLM. Tensor of shape `(num_videos, 3)`.
     * @param {Tensor} attention_mask (Optional) Mask to avoid performing attention on padding token indices. Tensor of shape `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`:
     * - 1 for tokens that are **not masked**,
     * - 0 for tokens that are **masked**.
     * @returns {[Tensor, Tensor]} [position_ids, mrope_position_deltas] with:
     * - position_ids: Tensor of shape `(3, batch_size, sequence_length)`.
     * - mrope_position_deltas: Tensor of shape `(batch_size)`.
     */
    get_rope_index(input_ids: Tensor, image_grid_thw: Tensor, video_grid_thw: Tensor, attention_mask: Tensor): [Tensor, Tensor];
    encode_image({ pixel_values, image_grid_thw }: {
        pixel_values: any;
        image_grid_thw: any;
    }): Promise<any>;
    _merge_input_ids_with_image_features(kwargs: any): {
        inputs_embeds: any;
        attention_mask: any;
    };
    prepare_inputs_for_generation(input_ids: any, model_inputs: any, generation_config: any): any;
}
import { PreTrainedModel } from '../modeling_utils.js';
import { Tensor } from '../../utils/tensor.js';
//# sourceMappingURL=modeling_qwen2_vl.d.ts.map