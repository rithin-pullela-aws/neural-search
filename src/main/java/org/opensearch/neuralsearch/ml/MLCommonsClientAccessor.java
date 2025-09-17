/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.ml;

import static org.opensearch.neuralsearch.processor.TextImageEmbeddingProcessor.INPUT_IMAGE;
import static org.opensearch.neuralsearch.processor.TextImageEmbeddingProcessor.INPUT_TEXT;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import org.opensearch.common.Nullable;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.util.CollectionUtils;
import org.opensearch.ml.client.MachineLearningNodeClient;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.dataset.MLInputDataset;
import org.opensearch.ml.common.dataset.TextDocsInputDataSet;
import org.opensearch.ml.common.dataset.TextSimilarityInputDataSet;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.execute.agent.AgentMLInput;
import org.opensearch.ml.common.input.parameter.MLAlgoParams;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.common.output.model.ModelResultFilter;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.neuralsearch.processor.InferenceRequest;
import org.opensearch.neuralsearch.processor.MapInferenceRequest;
import org.opensearch.neuralsearch.processor.SimilarityInferenceRequest;
import org.opensearch.neuralsearch.processor.TextInferenceRequest;
import org.opensearch.neuralsearch.util.RetryUtil;
import org.opensearch.ml.common.dataset.QuestionAnsweringInputDataSet;
import org.opensearch.neuralsearch.processor.highlight.SentenceHighlightingRequest;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;

/**
 * This class will act as an abstraction on the MLCommons client for accessing the ML Capabilities
 */
@Log4j2
public class MLCommonsClientAccessor {
    public final MachineLearningNodeClient mlClient;
    private final NamedXContentRegistry xContentRegistry;

    public MLCommonsClientAccessor(MachineLearningNodeClient mlClient, NamedXContentRegistry xContentRegistry) {
        this.mlClient = mlClient;
        this.xContentRegistry = xContentRegistry;
    }

    /**
     * Wrapper around {@link #inferenceSentences} that expected a single input text and produces a single floating
     * point vector as a response.
     *
     * @param modelId   {@link String}
     * @param inputText {@link String}
     * @param listener  {@link ActionListener} which will be called when prediction is completed or errored out
     */
    public void inferenceSentence(
        @NonNull final String modelId,
        @NonNull final String inputText,
        @NonNull final ActionListener<List<Number>> listener
    ) {

        inferenceSentences(
            TextInferenceRequest.builder().modelId(modelId).inputTexts(List.of(inputText)).build(),
            ActionListener.wrap(response -> {
                if (response.size() != 1) {
                    listener.onFailure(
                        new IllegalStateException(
                            "Unexpected number of vectors produced. Expected 1 vector to be returned, but got [" + response.size() + "]"
                        )
                    );
                    return;
                }

                listener.onResponse(response.getFirst());
            }, listener::onFailure)
        );
    }

    /**
     * Abstraction to call predict function of api of MLClient with provided targetResponse filters. It uses the
     * custom model provided as modelId and run the {@link FunctionName#TEXT_EMBEDDING}. The return will be sent
     * using the actionListener which will have a {@link List} of {@link List} of {@link Float} in the order of
     * inputText. We are not making this function generic enough to take any function or TaskType as currently we
     * need to run only TextEmbedding tasks only.
     *
     * @param inferenceRequest {@link InferenceRequest}
     * @param listener         {@link ActionListener} which will be called when prediction is completed or errored out.
     */
    public void inferenceSentences(
        @NonNull final TextInferenceRequest inferenceRequest,
        @NonNull final ActionListener<List<List<Number>>> listener
    ) {
        retryableInference(
            inferenceRequest,
            0,
            () -> createMLTextInput(inferenceRequest.getTargetResponseFilters(), inferenceRequest.getInputTexts()),
            this::buildVectorFromResponse,
            listener
        );
    }

    public void inferenceSentencesWithMapResult(
        @NonNull final TextInferenceRequest inferenceRequest,
        @Nullable MLAlgoParams mlAlgoParams,
        @NonNull final ActionListener<List<Map<String, ?>>> listener
    ) {
        retryableInference(inferenceRequest, 0, () -> {
            MLInput input = createMLTextInput(null, inferenceRequest.getInputTexts());
            if (mlAlgoParams != null) {
                input.setParameters(mlAlgoParams);
            }
            return input;
        }, this::buildMapResultFromResponse, listener);
    }

    /**
     * Abstraction to call predict function of api of MLClient with provided targetResponse filters. It uses the
     * custom model provided as modelId and run the {@link FunctionName#TEXT_EMBEDDING}. The return will be sent
     * using the actionListener which will have a list of floats in the order of inputText.
     *
     * @param inferenceRequest {@link InferenceRequest}
     * @param listener         {@link ActionListener} which will be called when prediction is completed or errored out.
     */
    public void inferenceSentencesMap(@NonNull MapInferenceRequest inferenceRequest, @NonNull final ActionListener<List<Number>> listener) {
        retryableInference(
            inferenceRequest,
            0,
            () -> createMLMultimodalInput(inferenceRequest.getTargetResponseFilters(), inferenceRequest.getInputObjects()),
            this::buildSingleVectorFromResponse,
            listener
        );
    }

    /**
     * Abstraction to call predict function of api of MLClient. It uses the custom model provided as modelId and the
     * {@link FunctionName#TEXT_SIMILARITY}. The return will be sent via actionListener as a list of floats representing
     * the similarity scores of the texts w.r.t. the query text, in the order of the input texts.
     *
     * @param inferenceRequest {@link InferenceRequest}
     * @param listener         {@link ActionListener} receives the result of the inference
     */
    public void inferenceSimilarity(
        @NonNull SimilarityInferenceRequest inferenceRequest,
        @NonNull final ActionListener<List<Float>> listener
    ) {
        retryableInference(
            inferenceRequest,
            0,
            () -> createMLTextPairsInput(inferenceRequest.getQueryText(), inferenceRequest.getInputTexts()),
            (mlOutput) -> buildVectorFromResponse(mlOutput).stream().map(v -> v.getFirst().floatValue()).collect(Collectors.toList()),
            listener
        );
    }

    /**
     * A generic function to make retryable inference request.
     * It allows caller to specify functions to vend their MLInput and process MLOutput.
     *
     * @param inferenceRequest inference request
     * @param retryTime retry time
     * @param mlInputSupplier a supplier to vend MLInput
     * @param mlOutputBuilder a consumer to consume MLOutput and provide processed output format.
     * @param listener a callback to handle result or failures.
     * @param <T> type of processed MLOutput format.
     */
    private <T> void retryableInference(
        final InferenceRequest inferenceRequest,
        final int retryTime,
        final Supplier<MLInput> mlInputSupplier,
        final Function<MLOutput, T> mlOutputBuilder,
        final ActionListener<T> listener
    ) {
        MLInput mlInput = mlInputSupplier.get();
        mlClient.predict(inferenceRequest.getModelId(), mlInput, ActionListener.wrap(mlOutput -> {
            final T result = mlOutputBuilder.apply(mlOutput);
            listener.onResponse(result);
        },
            e -> RetryUtil.handleRetryOrFailure(
                e,
                retryTime,
                () -> retryableInference(inferenceRequest, retryTime + 1, mlInputSupplier, mlOutputBuilder, listener),
                listener
            )
        ));
    }

    private MLInput createMLTextInput(final List<String> targetResponseFilters, List<String> inputText) {
        final ModelResultFilter modelResultFilter = new ModelResultFilter(false, true, targetResponseFilters, null);
        final MLInputDataset inputDataset = new TextDocsInputDataSet(inputText, modelResultFilter);
        return new MLInput(FunctionName.TEXT_EMBEDDING, null, inputDataset);
    }

    private MLInput createMLTextPairsInput(final String query, final List<String> inputText) {
        final MLInputDataset inputDataset = new TextSimilarityInputDataSet(query, inputText);
        return new MLInput(FunctionName.TEXT_SIMILARITY, null, inputDataset);
    }

    private <T extends Number> List<List<T>> buildVectorFromResponse(MLOutput mlOutput) {
        final List<List<T>> vector = new ArrayList<>();
        final ModelTensorOutput modelTensorOutput = (ModelTensorOutput) mlOutput;
        final List<ModelTensors> tensorOutputList = modelTensorOutput.getMlModelOutputs();
        for (final ModelTensors tensors : tensorOutputList) {
            final List<ModelTensor> tensorsList = tensors.getMlModelTensors();
            for (final ModelTensor tensor : tensorsList) {
                @SuppressWarnings("unchecked")
                List<T> tensorData = Arrays.stream(tensor.getData()).map(value -> (T) value).collect(Collectors.toList());
                vector.add(tensorData);
            }
        }
        return vector;
    }

    private List<Map<String, ?>> buildMapResultFromResponse(MLOutput mlOutput) {
        final ModelTensorOutput modelTensorOutput = (ModelTensorOutput) mlOutput;
        final List<ModelTensors> tensorOutputList = modelTensorOutput.getMlModelOutputs();
        if (CollectionUtils.isEmpty(tensorOutputList) || CollectionUtils.isEmpty(tensorOutputList.get(0).getMlModelTensors())) {
            throw new IllegalStateException(
                "Empty model result produced. Expected at least [1] tensor output and [1] model tensor, but got [0]"
            );
        }
        List<Map<String, ?>> resultMaps = new ArrayList<>();
        for (ModelTensors tensors : tensorOutputList) {
            List<ModelTensor> tensorList = tensors.getMlModelTensors();
            for (ModelTensor tensor : tensorList) {
                resultMaps.add(tensor.getDataAsMap());
            }
        }
        return resultMaps;
    }

    private String buildQueryResultFromResponseOfOutput(MLOutput mlOutput) {
        if (!(mlOutput instanceof ModelTensorOutput)) {
            throw new IllegalStateException("Expected ModelTensorOutput but got: " + mlOutput.getClass().getSimpleName());
        }
        final ModelTensorOutput modelTensorOutput = (ModelTensorOutput) mlOutput;

        final List<ModelTensors> tensorOutputList = modelTensorOutput.getMlModelOutputs();
        if (CollectionUtils.isEmpty(tensorOutputList)) {
            throw new IllegalStateException("Empty model result produced. Expected at least [1] tensor output, but got [0]");
        }

        // Iterate through all ModelTensors to find the DSL result
        for (ModelTensors tensors : tensorOutputList) {
            List<ModelTensor> tensorList = tensors.getMlModelTensors();
            if (!CollectionUtils.isEmpty(tensorList)) {
                for (ModelTensor tensor : tensorList) {
                    String result = tensor.getResult();
                    if (result != null && !result.trim().isEmpty()) {
                        return result;
                    }
                }
            }
        }

        throw new IllegalStateException("No valid DSL result found in model output");
    }

    private <T extends Number> List<T> buildSingleVectorFromResponse(final MLOutput mlOutput) {
        final List<List<T>> vector = buildVectorFromResponse(mlOutput);
        return vector.isEmpty() ? new ArrayList<>() : vector.get(0);
    }

    /**
     * Process the highlighting output from ML model response.
     * Converts the model output into a list of maps containing highlighting information.
     */
    private List<Map<String, Object>> processHighlightingOutput(ModelTensorOutput modelTensorOutput) {
        List<Map<String, Object>> results = new ArrayList<>();

        try {
            final List<ModelTensors> tensorOutputList = modelTensorOutput.getMlModelOutputs();

            if (CollectionUtils.isEmpty(tensorOutputList)) {
                return results;
            }

            for (ModelTensors tensors : tensorOutputList) {
                List<ModelTensor> tensorsList = tensors.getMlModelTensors();

                if (CollectionUtils.isEmpty(tensorsList)) {
                    log.warn("No tensors in model output");
                    continue;
                }

                // Process each tensor in the output
                for (ModelTensor tensor : tensorsList) {
                    Map<String, ?> dataMap = tensor.getDataAsMap(); // it stored in "result" in string type
                    if (dataMap != null && !dataMap.isEmpty()) {
                        // Cast the map to Map<String, Object> - this is safe as we're only reading from it
                        @SuppressWarnings("unchecked")
                        Map<String, Object> typedDataMap = (Map<String, Object>) dataMap;
                        results.add(typedDataMap);
                    }
                }
            }

            // If no results were found, add an empty map to maintain consistent response format
            if (results.isEmpty()) {
                results.add(Collections.emptyMap());
            }

            return results;
        } catch (Exception e) {
            throw new IllegalStateException("Error processing sentence highlighting output", e);
        }
    }

    private MLInput createMLMultimodalInput(final List<String> targetResponseFilters, final Map<String, String> input) {
        List<String> inputText = new ArrayList<>();
        inputText.add(input.get(INPUT_TEXT));
        if (input.containsKey(INPUT_IMAGE)) {
            inputText.add(input.get(INPUT_IMAGE));
        }
        final ModelResultFilter modelResultFilter = new ModelResultFilter(false, true, targetResponseFilters, null);
        final MLInputDataset inputDataset = new TextDocsInputDataSet(inputText, modelResultFilter);
        return new MLInput(FunctionName.TEXT_EMBEDDING, null, inputDataset);
    }

    public void getModel(@NonNull final String modelId, @NonNull final ActionListener<MLModel> listener) {
        retryableGetModel(modelId, 0, listener);
    }

    /**
     * Get model info for multiple model ids. It will send multiple getModel requests to get the model info in parallel.
     * It will fail if any one of the get model request fail. Only return the success result if all model info is
     * successfully retrieved.
     *
     * @param modelIds a set of model ids
     * @param onSuccess onSuccess consumer
     * @param onFailure onFailure consumer
     */
    public void getModels(
        @NonNull final Set<String> modelIds,
        @NonNull final Consumer<Map<String, MLModel>> onSuccess,
        @NonNull final Consumer<Exception> onFailure
    ) {
        if (modelIds.isEmpty()) {
            try {
                onSuccess.accept(Collections.emptyMap());
            } catch (Exception e) {
                onFailure.accept(e);
            }
            return;
        }

        final Map<String, MLModel> modelMap = new ConcurrentHashMap<>();
        final AtomicInteger counter = new AtomicInteger(modelIds.size());
        final AtomicBoolean hasError = new AtomicBoolean(false);
        final List<String> errors = Collections.synchronizedList(new ArrayList<>());

        for (String modelId : modelIds) {
            try {
                getModel(modelId, ActionListener.wrap(model -> {
                    modelMap.put(modelId, model);
                    if (counter.decrementAndGet() == 0) {
                        if (hasError.get()) {
                            onFailure.accept(new RuntimeException(String.join(";", errors)));
                        } else {
                            try {
                                onSuccess.accept(modelMap);
                            } catch (Exception e) {
                                onFailure.accept(e);
                            }
                        }
                    }
                }, e -> { handleGetModelException(hasError, errors, modelId, e, counter, onFailure); }));
            } catch (Exception e) {
                handleGetModelException(hasError, errors, modelId, e, counter, onFailure);
            }
        }

    }

    private void handleGetModelException(
        AtomicBoolean hasError,
        List<String> errors,
        String modelId,
        Exception e,
        AtomicInteger counter,
        @NonNull Consumer<Exception> onFailure
    ) {
        hasError.set(true);
        errors.add("Failed to fetch model [" + modelId + "]: " + e.getMessage());
        if (counter.decrementAndGet() == 0) {
            onFailure.accept(new RuntimeException(String.join(";", errors)));
        }
    }

    private void retryableGetModel(@NonNull final String modelId, final int retryTime, @NonNull final ActionListener<MLModel> listener) {
        mlClient.getModel(
            modelId,
            null,
            ActionListener.wrap(
                listener::onResponse,
                e -> RetryUtil.handleRetryOrFailure(e, retryTime, () -> retryableGetModel(modelId, retryTime + 1, listener), listener)
            )
        );
    }

    private void retryableGetAgent(
        @NonNull final String agentId,
        final int retryTime,
        @NonNull final ActionListener<Map<String, Object>> listener
    ) {
        // TODO: Update this method when MLAgentGetRequest and MLAgentGetAction become available
        // The preferred implementation would be:
        // MLAgentGetRequest mlAgentGetRequest = MLAgentGetRequest.builder()
        // .agentId(agentId)
        // .isUserInitiatedGetRequest(true)
        // .tenantId(null) // Optional
        // .build();
        // mlClient.execute(MLAgentGetAction.INSTANCE, mlAgentGetRequest, ActionListener.wrap(response -> {
        // MLAgent agent = response.getMlAgent();
        // Map<String, Object> agentInfo = convertMLAgentToMap(agent);
        // listener.onResponse(agentInfo);
        // }, e -> RetryUtil.handleRetryOrFailure(e, retryTime, () -> retryableGetAgent(agentId, retryTime + 1, listener), listener)));

        // For now, we'll use a generic approach that tries to get the agent as a model
        // and then parse the response to extract agent information
        mlClient.getModel(agentId, null, ActionListener.wrap(mlModel -> {
            // Convert MLModel to a Map representation for agent type detection
            Map<String, Object> agentInfo = convertMLModelToAgentInfo(mlModel);
            listener.onResponse(agentInfo);
        }, e -> RetryUtil.handleRetryOrFailure(e, retryTime, () -> retryableGetAgent(agentId, retryTime + 1, listener), listener)));
    }

    /**
     * Convert MLModel to a Map representation for agent type detection.
     * This is a temporary solution until MLAgentGetRequest is available.
     *
     * @param mlModel the ML model
     * @return a map representation of the agent info
     */
    private Map<String, Object> convertMLModelToAgentInfo(MLModel mlModel) {
        Map<String, Object> agentInfo = new HashMap<>();
        agentInfo.put("name", mlModel.getName());

        // Try to extract type information from model name or config
        String modelName = mlModel.getName();
        if (modelName != null) {
            String lowerName = modelName.toLowerCase();
            if (lowerName.contains("conversational") || lowerName.contains("conversation")) {
                agentInfo.put("type", "conversational");
            } else if (lowerName.contains("flow")) {
                agentInfo.put("type", "flow");
            } else {
                agentInfo.put("type", "flow"); // Default
            }
        } else {
            agentInfo.put("type", "flow"); // Default
        }

        return agentInfo;
    }

    /**
     * Performs sentence highlighting inference using the provided model.
     * This method will highlight relevant sentences in the context based on the question.
     *
     * @param inferenceRequest the request containing the question and context for highlighting
     * @param listener the listener to be called with the highlighting results
     */
    public void inferenceSentenceHighlighting(
        @NonNull final SentenceHighlightingRequest inferenceRequest,
        @NonNull final ActionListener<List<Map<String, Object>>> listener
    ) {
        retryableInference(inferenceRequest, 0, () -> {
            MLInputDataset inputDataset = new QuestionAnsweringInputDataSet(inferenceRequest.getQuestion(), inferenceRequest.getContext());
            return new MLInput(FunctionName.QUESTION_ANSWERING, null, inputDataset);
        }, (mlOutput) -> processHighlightingOutput((ModelTensorOutput) mlOutput), listener);
    }

    /**
     * Execute agent with provided parameters and return DSL query string.
     * This is a placeholder method that will get agent info and route to appropriate execution method.
     * Currently defaults to executeConversational for backward compatibility.
     *
     * @param agentId    the agent ID to execute
     * @param parameters the parameters to pass to the agent
     * @param listener   the listener to be called with the DSL query result
     */
    public void executeAgent(
        @NonNull final String agentId,
        @NonNull final Map<String, String> parameters,
        @NonNull final ActionListener<String> listener
    ) {
        // TODO: Implement proper agent type detection when MLAgentGetRequest becomes available
        // For now, we'll try to get agent info and route accordingly
        getAgent(agentId, ActionListener.wrap(agentInfo -> {
            String agentType = determineAgentTypeFromResponse(agentInfo);
            if ("flow".equals(agentType)) {
                executeFlow(agentId, parameters, listener);
            } else if ("conversational".equals(agentType)) {
                executeConversational(agentId, parameters, listener);
            } else {
                // Default to conversational for now since most agents are conversational
                executeConversational(agentId, parameters, listener);
            }
        }, e -> {
            // If getAgent fails, default to conversational
            log.warn("Failed to get agent info for [{}], defaulting to conversational execution: {}", agentId, e.getMessage());
            executeConversational(agentId, parameters, listener);
        }));
    }

    /**
     * Get agent information using the ML client.
     * This method calls the ML client to get agent details.
     *
     * @param agentId  the agent ID to get
     * @param listener the listener to be called with the agent information
     */
    public void getAgent(@NonNull final String agentId, @NonNull final ActionListener<Map<String, Object>> listener) {
        retryableGetAgent(agentId, 0, listener);
    }

    /**
     * Determine agent type from agent response JSON.
     * This method parses the agent response to extract the type field.
     *
     * @param agentInfo the agent information map
     * @return the agent type ("flow" or "conversational")
     */
    private String determineAgentTypeFromResponse(Map<String, Object> agentInfo) {
        if (agentInfo == null) {
            return "flow"; // Default to flow for backward compatibility
        }

        // Check for the type field in the agent response
        Object typeObj = agentInfo.get("type");
        if (typeObj instanceof String) {
            String type = (String) typeObj;
            if ("conversational".equals(type) || "conversation".equals(type)) {
                return "conversational";
            }
            if ("flow".equals(type)) {
                return "flow";
            }
        }

        // Check name for type indicators as fallback
        Object nameObj = agentInfo.get("name");
        if (nameObj instanceof String) {
            String name = ((String) nameObj).toLowerCase();
            if (name.contains("conversational") || name.contains("conversation")) {
                return "conversational";
            }
            if (name.contains("flow")) {
                return "flow";
            }
        }

        // Default to flow for backward compatibility
        return "flow";
    }

    /**
     * Execute flow-type agent with simple processing.
     * This method handles agents that return simple DSL queries directly.
     * Use this for agents that don't need complex JSON parsing.
     *
     * @param agentId    the agent ID to execute
     * @param parameters the parameters to pass to the agent
     * @param listener   the listener to be called with the DSL query result
     */
    public void executeFlow(
        @NonNull final String agentId,
        @NonNull final Map<String, String> parameters,
        @NonNull final ActionListener<String> listener
    ) {
        retryableExecuteFlow(agentId, parameters, 0, listener);
    }

    /**
     * Execute conversational-type agent with complex parsing logic.
     * This method handles agents that return complex JSON responses with multiple fields.
     * It parses the response to extract the query field and handles various response formats.
     * Use this for conversational agents that return structured JSON responses.
     *
     * @param agentId    the agent ID to execute
     * @param parameters the parameters to pass to the agent
     * @param listener   the listener to be called with the DSL query result
     */
    public void executeConversational(
        @NonNull final String agentId,
        @NonNull final Map<String, String> parameters,
        @NonNull final ActionListener<String> listener
    ) {
        retryableExecuteConversational(agentId, parameters, 0, listener);
    }

    private void retryableExecuteFlow(
        final String agentId,
        final Map<String, String> parameters,
        final int retryTime,
        final ActionListener<String> listener
    ) {
        RemoteInferenceInputDataSet dataset = RemoteInferenceInputDataSet.builder().parameters(parameters).build();
        AgentMLInput agentMLInput = new AgentMLInput(agentId, null, FunctionName.AGENT, dataset);
        mlClient.execute(FunctionName.AGENT, agentMLInput, ActionListener.wrap(response -> {
            try {
                // Extract DSL query from inference results following the structure:
                MLOutput mlOutput = (MLOutput) response.getOutput();
                final String inferenceResults = buildQueryResultFromResponseOfOutput(mlOutput);

                listener.onResponse(inferenceResults);
            } catch (Exception e) {
                listener.onFailure(new IllegalStateException("Failed to extract result from agent response", e));
            }
        },
            e -> RetryUtil.handleRetryOrFailure(
                e,
                retryTime,
                () -> retryableExecuteFlow(agentId, parameters, retryTime + 1, listener),
                listener
            )
        ));
    }

    private void retryableExecuteConversational(
        final String agentId,
        final Map<String, String> parameters,
        final int retryTime,
        final ActionListener<String> listener
    ) {
        // Log parameters sent to agent
        log.info("=== SENDING PARAMETERS TO AGENT ===");
        log.info("Agent ID: [{}]", agentId);
        log.info("Parameters: [{}]", parameters);
        log.info("=== END PARAMETERS ===");

        RemoteInferenceInputDataSet dataset = RemoteInferenceInputDataSet.builder().parameters(parameters).build();
        AgentMLInput agentMLInput = new AgentMLInput(agentId, null, FunctionName.AGENT, dataset);
        mlClient.execute(FunctionName.AGENT, agentMLInput, ActionListener.wrap(response -> {
            try {
                ModelTensorOutput output = (ModelTensorOutput) response.getOutput();
                if (output == null) {
                    throw new IllegalStateException("Null response from agent");
                }

                // Extract the response from the ModelTensorOutput
                String agentResponse = extractResponseFromOutput(output);
                if (agentResponse == null) {
                    throw new IllegalStateException("No response found in agent output");
                }

                // Parse the agent response - handle both JSON and plain string responses
                String generatedQuery = parseConversationalResponse(agentResponse);
                listener.onResponse(generatedQuery);
            } catch (Exception e) {
                listener.onFailure(new IllegalStateException("Failed to process conversational agent response", e));
            }
        },
            e -> RetryUtil.handleRetryOrFailure(
                e,
                retryTime,
                () -> retryableExecuteConversational(agentId, parameters, retryTime + 1, listener),
                listener
            )
        ));
    }

    /**
     * Extract response from ModelTensorOutput for conversational agents.
     * This method handles the complex response extraction logic.
     *
     * @param output the ModelTensorOutput from the agent
     * @return the extracted response string
     */
    private String extractResponseFromOutput(ModelTensorOutput output) {
        if (output == null || output.getMlModelOutputs() == null || output.getMlModelOutputs().isEmpty()) {
            return null;
        }

        for (ModelTensors tensors : output.getMlModelOutputs()) {
            if (tensors.getMlModelTensors() != null) {
                for (ModelTensor tensor : tensors.getMlModelTensors()) {
                    if ("response".equals(tensor.getName())) {
                        Map<String, ?> dataMap = tensor.getDataAsMap();
                        if (dataMap != null && dataMap.containsKey("response")) {
                            return (String) dataMap.get("response");
                        }
                        return tensor.getResult();
                    }
                }
            }
        }

        return null;
    }

    /**
     * Parse conversational agent response to extract the generated query.
     * This method handles the complex JSON parsing logic from the processor.
     *
     * @param agentResponse the raw response from the conversational agent
     * @return the extracted query string
     */
    private String parseConversationalResponse(String agentResponse) {
        if (agentResponse == null || agentResponse.trim().isEmpty()) {
            throw new IllegalStateException("Empty agent response");
        }

        log.info("=== PARSING CONVERSATIONAL AGENT RESPONSE ===");
        log.info("Attempting to parse: [{}]", agentResponse);

        String generatedQuery = null;
        String stepsByAgent = null;

        // First, try to parse as JSON
        try {
            BytesReference bytes = new BytesArray(agentResponse);
            try (XContentParser parser = XContentType.JSON.xContent().createParser(xContentRegistry, null, bytes.streamInput())) {
                if (parser.currentToken() == null) {
                    parser.nextToken();
                }

                log.info("Parser current token: [{}]", parser.currentToken());

                if (parser.currentToken() == XContentParser.Token.START_ARRAY) {
                    // Top-level is an array, pick the first element
                    log.info("Top-level response is an array. Parsing first element.");
                    if (parser.nextToken() == XContentParser.Token.START_OBJECT) {
                        // Parse first object in array
                        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
                            String fieldName = parser.currentName();
                            parser.nextToken();

                            log.info("[ARRAY] Found field: [{}] with token: [{}]", fieldName, parser.currentToken());

                            if ("dsl_query".equals(fieldName)) {
                                if (parser.currentToken() == XContentParser.Token.START_OBJECT) {
                                    generatedQuery = readCurrentObjectAsString(parser);
                                    log.info("[ARRAY] Extracted dsl_query object JSON");
                                } else {
                                    String queryString = parser.text();
                                    generatedQuery = queryString;
                                    log.info("[ARRAY] Extracted dsl_query string");
                                }
                            } else if ("agent_steps_summary".equals(fieldName)
                                || "agent_summary".equals(fieldName)
                                || "steps_by_agent".equals(fieldName)) {
                                    stepsByAgent = parser.currentToken() == XContentParser.Token.VALUE_STRING ? parser.text() : null;
                                    log.info("[ARRAY] Extracted steps summary field: [{}]", stepsByAgent);
                                } else if ("generated_query".equals(fieldName) || "query".equals(fieldName)) {
                                    // Fallback fields
                                    if (parser.currentToken() == XContentParser.Token.START_OBJECT) {
                                        generatedQuery = readCurrentObjectAsString(parser);
                                    } else {
                                        generatedQuery = parser.text();
                                    }
                                } else {
                                    parser.skipChildren();
                                }
                        }
                    }

                    // consume remaining array tokens if any
                    while (parser.currentToken() != XContentParser.Token.END_ARRAY
                        && parser.nextToken() != XContentParser.Token.END_ARRAY) {
                        parser.skipChildren();
                    }
                } else if (parser.currentToken() == XContentParser.Token.START_OBJECT) {
                    // It's a JSON object, parse it
                    while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
                        String fieldName = parser.currentName();
                        parser.nextToken();

                        log.info("Found field: [{}] with token: [{}]", fieldName, parser.currentToken());

                        if ("dsl_query".equals(fieldName)) {
                            if (parser.currentToken() == XContentParser.Token.START_OBJECT) {
                                generatedQuery = readCurrentObjectAsString(parser);
                                log.info("Extracted dsl_query object JSON");
                            } else {
                                String queryString = parser.text();
                                // Try to validate JSON, else keep as-is
                                try {
                                    BytesReference queryBytes = new BytesArray(queryString);
                                    try (
                                        XContentParser queryParser = XContentType.JSON.xContent()
                                            .createParser(xContentRegistry, null, queryBytes.streamInput())
                                    ) {
                                        if (queryParser.nextToken() != null) {
                                            generatedQuery = queryString;
                                            log.info("DSL query field contains valid JSON string");
                                        }
                                    }
                                } catch (Exception queryParseException) {
                                    log.warn(
                                        "DSL query field string is not valid JSON, using raw string: [{}]",
                                        queryParseException.getMessage()
                                    );
                                    generatedQuery = queryString;
                                }
                            }
                        } else if ("query".equals(fieldName)) {
                            String queryString = parser.text();
                            log.info("Found query field: [{}]", queryString);

                            // The query field contains escaped JSON, so we need to parse it again
                            try {
                                BytesReference queryBytes = new BytesArray(queryString);
                                try (
                                    XContentParser queryParser = XContentType.JSON.xContent()
                                        .createParser(xContentRegistry, null, queryBytes.streamInput())
                                ) {
                                    // Validate that it's valid JSON
                                    if (queryParser.nextToken() != null) {
                                        generatedQuery = queryString; // Use the original string for SearchSourceBuilder parsing
                                        log.info("Query field contains valid JSON: [{}]", generatedQuery);
                                    }
                                }
                            } catch (Exception queryParseException) {
                                log.warn("Query field is not valid JSON, treating as plain string: [{}]", queryParseException.getMessage());
                                generatedQuery = queryString;
                            }
                        } else if ("agent_steps_summary".equals(fieldName)) {
                            stepsByAgent = parser.text();
                            log.info("Found agent_steps_summary field: [{}]", stepsByAgent);
                        } else if ("agent_summary".equals(fieldName)) {
                            stepsByAgent = parser.text();
                            log.info("Found agent_summary field: [{}]", stepsByAgent);
                        } else if ("generated_query".equals(fieldName)) {
                            String queryString = parser.text();
                            log.info("Found generated_query field: [{}]", queryString);

                            // Handle escaped JSON in generated_query field as well
                            try {
                                BytesReference queryBytes = new BytesArray(queryString);
                                try (
                                    XContentParser queryParser = XContentType.JSON.xContent()
                                        .createParser(xContentRegistry, null, queryBytes.streamInput())
                                ) {
                                    if (queryParser.nextToken() != null) {
                                        generatedQuery = queryString;
                                        log.info("Generated query field contains valid JSON: [{}]", generatedQuery);
                                    }
                                }
                            } catch (Exception queryParseException) {
                                log.warn(
                                    "Generated query field is not valid JSON, treating as plain string: [{}]",
                                    queryParseException.getMessage()
                                );
                                generatedQuery = queryString;
                            }
                        } else if ("steps_by_agent".equals(fieldName)) {
                            stepsByAgent = parser.text();
                            log.info("Found steps_by_agent field: [{}]", stepsByAgent);
                        } else {
                            log.info("Skipping unknown field: [{}]", fieldName);
                            parser.skipChildren();
                        }
                    }
                } else if (parser.currentToken() == XContentParser.Token.VALUE_STRING) {
                    // It's a string - try to parse it as JSON directly
                    String stringValue = parser.text();
                    log.info("Found string response: [{}]", stringValue);

                    // Try to parse the string as JSON
                    try {
                        BytesReference jsonBytes = new BytesArray(stringValue);
                        try (
                            XContentParser jsonParser = XContentType.JSON.xContent()
                                .createParser(xContentRegistry, null, jsonBytes.streamInput())
                        ) {
                            if (jsonParser.currentToken() == null) {
                                jsonParser.nextToken();
                            }

                            log.info("String parser current token: [{}]", jsonParser.currentToken());

                            if (jsonParser.currentToken() == XContentParser.Token.START_OBJECT) {
                                // The string contains a JSON object, parse it
                                log.info("String contains JSON object, parsing...");
                                while (jsonParser.nextToken() != XContentParser.Token.END_OBJECT) {
                                    String fieldName = jsonParser.currentName();
                                    jsonParser.nextToken();

                                    log.info("Found JSON string field: [{}] with token: [{}]", fieldName, jsonParser.currentToken());

                                    if ("dsl_query".equals(fieldName)) {
                                        generatedQuery = jsonParser.text();
                                        log.info("Found dsl_query field in JSON string: [{}]", generatedQuery);
                                    } else if ("query".equals(fieldName)) {
                                        generatedQuery = jsonParser.text();
                                        log.info("Found query field in JSON string: [{}]", generatedQuery);
                                    } else if ("agent_steps_summary".equals(fieldName)) {
                                        stepsByAgent = jsonParser.text();
                                        log.info("Found agent_steps_summary field in JSON string: [{}]", stepsByAgent);
                                    } else if ("agent_summary".equals(fieldName)) {
                                        stepsByAgent = jsonParser.text();
                                        log.info("Found agent_summary field in JSON string: [{}]", stepsByAgent);
                                    } else if ("generated_query".equals(fieldName)) {
                                        generatedQuery = jsonParser.text();
                                        log.info("Found generated_query field in JSON string: [{}]", generatedQuery);
                                    } else if ("steps_by_agent".equals(fieldName)) {
                                        stepsByAgent = jsonParser.text();
                                        log.info("Found steps_by_agent field in JSON string: [{}]", stepsByAgent);
                                    } else {
                                        log.info("Skipping unknown field in JSON string: [{}]", fieldName);
                                        jsonParser.skipChildren();
                                    }
                                }
                            } else {
                                // The string doesn't contain a JSON object, treat it as a plain string
                                log.info("String does not contain JSON object, treating as plain string");
                                generatedQuery = stringValue;
                            }
                        }
                    } catch (Exception jsonStringParseException) {
                        log.info("Failed to parse string as JSON, treating as plain string: [{}]", jsonStringParseException.getMessage());
                        // If JSON string parsing fails, treat the entire response as a plain string
                        generatedQuery = stringValue;
                    }
                }
            }
        } catch (Exception jsonParseException) {
            log.info("Failed to parse as JSON, treating as plain string: [{}]", jsonParseException.getMessage());
            // If JSON parsing fails, treat the entire response as a plain string
            generatedQuery = agentResponse;
            log.info("Using entire response as generated query: [{}]", generatedQuery);
        }

        // Validate that we got the generated query
        if (generatedQuery == null || generatedQuery.trim().isEmpty()) {
            throw new IllegalStateException("No generated query found in conversational agent response");
        }

        log.info("=== EXTRACTED VALUES ===");
        log.info("Generated query: [{}]", generatedQuery);
        log.info("Steps by agent: [{}]", stepsByAgent);
        log.info("=== END EXTRACTED VALUES ===");

        // Log the extracted DSL query at info level for easy monitoring
        log.info("=== EXTRACTED DSL QUERY ===");
        log.info("DSL Query: [{}]", generatedQuery);
        log.info("=== END DSL QUERY ===");

        return generatedQuery;
    }

    private String readCurrentObjectAsString(final XContentParser parser) throws Exception {
        // parser is currently at START_OBJECT; copy the full object to builder
        XContentBuilder builder = XContentBuilder.builder(XContentType.JSON.xContent());
        builder.copyCurrentStructure(parser);
        return BytesReference.bytes(builder).utf8ToString();
    }
}
