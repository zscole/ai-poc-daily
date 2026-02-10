/**
 * @typedef {keyof typeof SUPPORTED_TASKS} TaskType
 * @typedef {keyof typeof TASK_ALIASES} AliasType
 * @typedef {TaskType | AliasType} PipelineType All possible pipeline types.
 * @typedef {{[K in TaskType]: InstanceType<typeof SUPPORTED_TASKS[K]["pipeline"]>}} SupportedTasks A mapping of pipeline names to their corresponding pipeline classes.
 * @typedef {{[K in AliasType]: InstanceType<typeof SUPPORTED_TASKS[TASK_ALIASES[K]]["pipeline"]>}} AliasTasks A mapping from pipeline aliases to their corresponding pipeline classes.
 * @typedef {SupportedTasks & AliasTasks} AllTasks A mapping from all pipeline names and aliases to their corresponding pipeline classes.
 */
/**
 * Utility factory method to build a `Pipeline` object.
 *
 * @template {PipelineType} T The type of pipeline to return.
 * @param {T} task The task defining which pipeline will be returned. Currently accepted tasks are:
 *  - `"audio-classification"`: will return a `AudioClassificationPipeline`.
 *  - `"automatic-speech-recognition"`: will return a `AutomaticSpeechRecognitionPipeline`.
 *  - `"depth-estimation"`: will return a `DepthEstimationPipeline`.
 *  - `"document-question-answering"`: will return a `DocumentQuestionAnsweringPipeline`.
 *  - `"feature-extraction"`: will return a `FeatureExtractionPipeline`.
 *  - `"fill-mask"`: will return a `FillMaskPipeline`.
 *  - `"image-classification"`: will return a `ImageClassificationPipeline`.
 *  - `"image-segmentation"`: will return a `ImageSegmentationPipeline`.
 *  - `"image-to-text"`: will return a `ImageToTextPipeline`.
 *  - `"object-detection"`: will return a `ObjectDetectionPipeline`.
 *  - `"question-answering"`: will return a `QuestionAnsweringPipeline`.
 *  - `"summarization"`: will return a `SummarizationPipeline`.
 *  - `"text2text-generation"`: will return a `Text2TextGenerationPipeline`.
 *  - `"text-classification"` (alias "sentiment-analysis" available): will return a `TextClassificationPipeline`.
 *  - `"text-generation"`: will return a `TextGenerationPipeline`.
 *  - `"token-classification"` (alias "ner" available): will return a `TokenClassificationPipeline`.
 *  - `"translation"`: will return a `TranslationPipeline`.
 *  - `"translation_xx_to_yy"`: will return a `TranslationPipeline`.
 *  - `"zero-shot-classification"`: will return a `ZeroShotClassificationPipeline`.
 *  - `"zero-shot-audio-classification"`: will return a `ZeroShotAudioClassificationPipeline`.
 *  - `"zero-shot-image-classification"`: will return a `ZeroShotImageClassificationPipeline`.
 *  - `"zero-shot-object-detection"`: will return a `ZeroShotObjectDetectionPipeline`.
 * @param {string} [model=null] The name of the pre-trained model to use. If not specified, the default model for the task will be used.
 * @param {import('./utils/hub.js').PretrainedModelOptions} [options] Optional parameters for the pipeline.
 * @returns {Promise<AllTasks[T]>} A Pipeline object for the specified task.
 * @throws {Error} If an unsupported pipeline is requested.
 */
export function pipeline<T extends PipelineType>(task: T, model?: string, { progress_callback, config, cache_dir, local_files_only, revision, device, dtype, subfolder, use_external_data_format, model_file_name, session_options, }?: import("./utils/hub.js").PretrainedModelOptions): Promise<AllTasks[T]>;
export type TaskType = keyof typeof SUPPORTED_TASKS;
export type AliasType = keyof typeof TASK_ALIASES;
/**
 * All possible pipeline types.
 */
export type PipelineType = TaskType | AliasType;
/**
 * A mapping of pipeline names to their corresponding pipeline classes.
 */
export type SupportedTasks = { [K in TaskType]: InstanceType<(typeof SUPPORTED_TASKS)[K]["pipeline"]>; };
/**
 * A mapping from pipeline aliases to their corresponding pipeline classes.
 */
export type AliasTasks = { [K in AliasType]: InstanceType<(typeof SUPPORTED_TASKS)[Readonly<{
    'sentiment-analysis': "text-classification";
    ner: "token-classification";
    asr: "automatic-speech-recognition";
    'text-to-speech': "text-to-audio";
    embeddings: "feature-extraction";
}>[K]]["pipeline"]>; };
/**
 * A mapping from all pipeline names and aliases to their corresponding pipeline classes.
 */
export type AllTasks = SupportedTasks & AliasTasks;
import { TextClassificationPipeline } from './pipelines/text-classification.js';
import { TokenClassificationPipeline } from './pipelines/token-classification.js';
import { QuestionAnsweringPipeline } from './pipelines/question-answering.js';
import { FillMaskPipeline } from './pipelines/fill-mask.js';
import { SummarizationPipeline } from './pipelines/summarization.js';
import { TranslationPipeline } from './pipelines/translation.js';
import { Text2TextGenerationPipeline } from './pipelines/text2text-generation.js';
import { TextGenerationPipeline } from './pipelines/text-generation.js';
import { ZeroShotClassificationPipeline } from './pipelines/zero-shot-classification.js';
import { AudioClassificationPipeline } from './pipelines/audio-classification.js';
import { ZeroShotAudioClassificationPipeline } from './pipelines/zero-shot-audio-classification.js';
import { AutomaticSpeechRecognitionPipeline } from './pipelines/automatic-speech-recognition.js';
import { TextToAudioPipeline } from './pipelines/text-to-audio.js';
import { ImageToTextPipeline } from './pipelines/image-to-text.js';
import { ImageClassificationPipeline } from './pipelines/image-classification.js';
import { ImageSegmentationPipeline } from './pipelines/image-segmentation.js';
import { BackgroundRemovalPipeline } from './pipelines/background-removal.js';
import { ZeroShotImageClassificationPipeline } from './pipelines/zero-shot-image-classification.js';
import { ObjectDetectionPipeline } from './pipelines/object-detection.js';
import { ZeroShotObjectDetectionPipeline } from './pipelines/zero-shot-object-detection.js';
import { DocumentQuestionAnsweringPipeline } from './pipelines/document-question-answering.js';
import { ImageToImagePipeline } from './pipelines/image-to-image.js';
import { DepthEstimationPipeline } from './pipelines/depth-estimation.js';
import { FeatureExtractionPipeline } from './pipelines/feature-extraction.js';
import { ImageFeatureExtractionPipeline } from './pipelines/image-feature-extraction.js';
declare const SUPPORTED_TASKS: Readonly<{
    'text-classification': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof TextClassificationPipeline;
        model: typeof AutoModelForSequenceClassification;
        default: {
            model: string;
        };
        type: string;
    };
    'token-classification': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof TokenClassificationPipeline;
        model: typeof AutoModelForTokenClassification;
        default: {
            model: string;
        };
        type: string;
    };
    'question-answering': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof QuestionAnsweringPipeline;
        model: typeof AutoModelForQuestionAnswering;
        default: {
            model: string;
        };
        type: string;
    };
    'fill-mask': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof FillMaskPipeline;
        model: typeof AutoModelForMaskedLM;
        default: {
            model: string;
            dtype: string;
        };
        type: string;
    };
    summarization: {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof SummarizationPipeline;
        model: typeof AutoModelForSeq2SeqLM;
        default: {
            model: string;
        };
        type: string;
    };
    translation: {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof TranslationPipeline;
        model: typeof AutoModelForSeq2SeqLM;
        default: {
            model: string;
        };
        type: string;
    };
    'text2text-generation': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof Text2TextGenerationPipeline;
        model: typeof AutoModelForSeq2SeqLM;
        default: {
            model: string;
        };
        type: string;
    };
    'text-generation': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof TextGenerationPipeline;
        model: typeof AutoModelForCausalLM;
        default: {
            model: string;
            dtype: string;
        };
        type: string;
    };
    'zero-shot-classification': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof ZeroShotClassificationPipeline;
        model: typeof AutoModelForSequenceClassification;
        default: {
            model: string;
        };
        type: string;
    };
    'audio-classification': {
        pipeline: typeof AudioClassificationPipeline;
        model: typeof AutoModelForAudioClassification;
        processor: typeof AutoProcessor;
        default: {
            model: string;
        };
        type: string;
    };
    'zero-shot-audio-classification': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof ZeroShotAudioClassificationPipeline;
        model: typeof AutoModel;
        processor: typeof AutoProcessor;
        default: {
            model: string;
        };
        type: string;
    };
    'automatic-speech-recognition': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof AutomaticSpeechRecognitionPipeline;
        model: (typeof AutoModelForSpeechSeq2Seq)[];
        processor: typeof AutoProcessor;
        default: {
            model: string;
        };
        type: string;
    };
    'text-to-audio': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof TextToAudioPipeline;
        model: (typeof AutoModelForTextToSpectrogram)[];
        processor: (typeof AutoProcessor)[];
        default: {
            model: string;
            dtype: string;
        };
        type: string;
    };
    'image-to-text': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof ImageToTextPipeline;
        model: typeof AutoModelForVision2Seq;
        processor: typeof AutoProcessor;
        default: {
            model: string;
        };
        type: string;
    };
    'image-classification': {
        pipeline: typeof ImageClassificationPipeline;
        model: typeof AutoModelForImageClassification;
        processor: typeof AutoProcessor;
        default: {
            model: string;
        };
        type: string;
    };
    'image-segmentation': {
        pipeline: typeof ImageSegmentationPipeline;
        model: (typeof AutoModelForImageSegmentation)[];
        processor: typeof AutoProcessor;
        default: {
            model: string;
        };
        type: string;
    };
    'background-removal': {
        pipeline: typeof BackgroundRemovalPipeline;
        model: (typeof AutoModelForImageSegmentation)[];
        processor: typeof AutoProcessor;
        default: {
            model: string;
        };
        type: string;
    };
    'zero-shot-image-classification': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof ZeroShotImageClassificationPipeline;
        model: typeof AutoModel;
        processor: typeof AutoProcessor;
        default: {
            model: string;
        };
        type: string;
    };
    'object-detection': {
        pipeline: typeof ObjectDetectionPipeline;
        model: typeof AutoModelForObjectDetection;
        processor: typeof AutoProcessor;
        default: {
            model: string;
        };
        type: string;
    };
    'zero-shot-object-detection': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof ZeroShotObjectDetectionPipeline;
        model: typeof AutoModelForZeroShotObjectDetection;
        processor: typeof AutoProcessor;
        default: {
            model: string;
        };
        type: string;
    };
    'document-question-answering': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof DocumentQuestionAnsweringPipeline;
        model: typeof AutoModelForDocumentQuestionAnswering;
        processor: typeof AutoProcessor;
        default: {
            model: string;
        };
        type: string;
    };
    'image-to-image': {
        pipeline: typeof ImageToImagePipeline;
        model: typeof AutoModelForImageToImage;
        processor: typeof AutoProcessor;
        default: {
            model: string;
        };
        type: string;
    };
    'depth-estimation': {
        pipeline: typeof DepthEstimationPipeline;
        model: typeof AutoModelForDepthEstimation;
        processor: typeof AutoProcessor;
        default: {
            model: string;
        };
        type: string;
    };
    'feature-extraction': {
        tokenizer: typeof AutoTokenizer;
        pipeline: typeof FeatureExtractionPipeline;
        model: typeof AutoModel;
        default: {
            model: string;
            dtype: string;
        };
        type: string;
    };
    'image-feature-extraction': {
        processor: typeof AutoProcessor;
        pipeline: typeof ImageFeatureExtractionPipeline;
        model: (typeof AutoModel)[];
        default: {
            model: string;
            dtype: string;
        };
        type: string;
    };
}>;
declare const TASK_ALIASES: Readonly<{
    'sentiment-analysis': "text-classification";
    ner: "token-classification";
    asr: "automatic-speech-recognition";
    'text-to-speech': "text-to-audio";
    embeddings: "feature-extraction";
}>;
import { AutoTokenizer } from './models/auto/tokenization_auto.js';
import { AutoModelForSequenceClassification } from './models/auto/modeling_auto.js';
import { AutoModelForTokenClassification } from './models/auto/modeling_auto.js';
import { AutoModelForQuestionAnswering } from './models/auto/modeling_auto.js';
import { AutoModelForMaskedLM } from './models/auto/modeling_auto.js';
import { AutoModelForSeq2SeqLM } from './models/auto/modeling_auto.js';
import { AutoModelForCausalLM } from './models/auto/modeling_auto.js';
import { AutoModelForAudioClassification } from './models/auto/modeling_auto.js';
import { AutoProcessor } from './models/auto/processing_auto.js';
import { AutoModel } from './models/auto/modeling_auto.js';
import { AutoModelForSpeechSeq2Seq } from './models/auto/modeling_auto.js';
import { AutoModelForTextToSpectrogram } from './models/auto/modeling_auto.js';
import { AutoModelForVision2Seq } from './models/auto/modeling_auto.js';
import { AutoModelForImageClassification } from './models/auto/modeling_auto.js';
import { AutoModelForImageSegmentation } from './models/auto/modeling_auto.js';
import { AutoModelForObjectDetection } from './models/auto/modeling_auto.js';
import { AutoModelForZeroShotObjectDetection } from './models/auto/modeling_auto.js';
import { AutoModelForDocumentQuestionAnswering } from './models/auto/modeling_auto.js';
import { AutoModelForImageToImage } from './models/auto/modeling_auto.js';
import { AutoModelForDepthEstimation } from './models/auto/modeling_auto.js';
export { TextClassificationPipeline, TokenClassificationPipeline, QuestionAnsweringPipeline, FillMaskPipeline, SummarizationPipeline, TranslationPipeline, Text2TextGenerationPipeline, TextGenerationPipeline, ZeroShotClassificationPipeline, AudioClassificationPipeline, ZeroShotAudioClassificationPipeline, AutomaticSpeechRecognitionPipeline, TextToAudioPipeline, ImageToTextPipeline, ImageClassificationPipeline, ImageSegmentationPipeline, BackgroundRemovalPipeline, ZeroShotImageClassificationPipeline, ObjectDetectionPipeline, ZeroShotObjectDetectionPipeline, DocumentQuestionAnsweringPipeline, ImageToImagePipeline, DepthEstimationPipeline, FeatureExtractionPipeline, ImageFeatureExtractionPipeline };
//# sourceMappingURL=pipelines.d.ts.map