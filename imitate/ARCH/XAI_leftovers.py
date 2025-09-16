# # %% ==========================================================================
# # Integrated Gradients for interpretability on output layer
# # =============================================================================
# from captum.attr import IntegratedGradients

# # Convert test data to tensor
# X_test_tensor = torch.FloatTensor(X_test)

# # Create a wrapper class that inherits from nn.Module
# class ModelWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
        
#     def forward(self, input_data):
#         features = self.model.policy.features_extractor(input_data)
#         pi_features, _ = self.model.policy.mlp_extractor(features)
#         return self.model.policy.action_net(pi_features)

# # Create model wrapper instance
# model_wrapper = ModelWrapper(model)

# # Initialize IntegratedGradients
# ig = IntegratedGradients(model_wrapper)

# # Calculate attributions
# attributions = ig.attribute(X_test_tensor)

# # Visualize the attribution analysis
# plt.figure(figsize=(8, 6))
# plt.imshow(attributions.detach().numpy(), cmap="seismic", aspect="auto")
# plt.colorbar()
# plt.xlabel("Features")
# plt.ylabel("Samples")
# plt.title("Integrated Gradients Analysis of BC Model Predictions")
# plt.show()

# # %% ==========================================================================
# # Correlation neuron activity with feature levels
# # =============================================================================
# def analyze_neuron_feature_correlations(model, layer_name, input_tensor):
#     """
#     Create a correlation heatmap between input features and neuron activations.
    
#     Args:
#         model: The trained BC model
#         layer_name: Name of the layer to analyze
#         input_tensor: Input features tensor
#     """
#     class LayerWrapper(torch.nn.Module):
#         def __init__(self, model, layer_name):
#             super().__init__()
#             self.model = model
#             self.layer_name = layer_name
            
#         def forward(self, x):
#             if not isinstance(x, torch.Tensor):
#                 x = torch.tensor(x, dtype=torch.float32)
            
#             features = self.model.policy.features_extractor(x)
            
#             if self.layer_name == 'pi_features':
#                 pi_features, _ = self.model.policy.mlp_extractor(features)
#                 return pi_features
#             elif self.layer_name == 'vf_features':
#                 _, vf_features = self.model.policy.mlp_extractor(features)
#                 return vf_features
#             elif self.layer_name == 'action_net':
#                 pi_features, _ = self.model.policy.mlp_extractor(features)
#                 return self.model.policy.action_net(pi_features)
#             elif self.layer_name == 'value_net':
#                 _, vf_features = self.model.policy.mlp_extractor(features)
#                 return self.model.policy.value_net(vf_features)
    
#     # Get neuron activations
#     wrapped_model = LayerWrapper(model, layer_name)
#     with torch.no_grad():
#         activations = wrapped_model(input_tensor).numpy()
    
#     # Convert input tensor to numpy
#     features_np = input_tensor.numpy()
    
#     # Calculate correlations
#     correlations = np.zeros((activations.shape[1], features_np.shape[1]))
#     for i in range(activations.shape[1]):  # for each neuron
#         for j in range(features_np.shape[1]):  # for each feature
#             correlations[i,j] = np.corrcoef(activations[:,i], features_np[:,j])[0,1]
    
#     # Plot correlation heatmap
#     plt.figure(figsize=(12, 8))
#     im = plt.imshow(correlations, aspect='auto', cmap='RdBu_r')
#     plt.colorbar(im, label='Correlation Coefficient')
    
#     # Add labels
#     plt.title(f'Neuron-Feature Correlations in {layer_name} Layer')
#     plt.xlabel('Input Features')
#     plt.ylabel('Neuron Index')
    
#     # Add feature names
#     plt.xticks(range(len(input_features)), input_features, rotation=45, ha='right')
#     plt.yticks(range(activations.shape[1]), [f'N{i}' for i in range(activations.shape[1])])
    
#     plt.tight_layout()
#     plt.show()
    
#     return correlations

# layer_names = [
#     'pi_features', 
#     'vf_features', 
#     'action_net', 
#     'value_net'
#     ]

# # Example usage
# correlations = analyze_neuron_feature_correlations(
#     model,
#     layer_names[0],  # 'pi_features'
#     torch.tensor(X_test, dtype=torch.float32)
# )

# # Linear feature associations
# feature_corr_sum = [np.sum(correlations[:,i]) for i in range(len(input_features))]

# # plt.figure(figsize=(12, 6))
# # plt.bar(input_features, feature_corr_sum)
# # plt.xticks(range(len(input_features)), input_features, rotation=45, ha='right')
# # plt.xlabel('Features')
# # plt.ylabel('Sum of Neuron Correlations')
# # plt.title('Total Correlation with Neuron Activity')


# # %% ==========================================================================
# # Feature attributions by manually computing gradients
# # =============================================================================
# def analyze_layer_attributions(model, layer_name, input_tensor):
#     """
#     Analyze feature importance by computing mean attributions for each feature.
    
#     Args:
#         model: The trained BC model
#         layer_name: Name of the layer to analyze
#         input_tensor: Input features tensor
#         features: List of feature names
#     """
#     class LayerWrapper(torch.nn.Module):
#         def __init__(self, model):
#             super().__init__()
#             self.model = model
            
#         def forward(self, x):
#             if not isinstance(x, torch.Tensor):
#                 x = torch.tensor(x, dtype=torch.float32)
            
#             features = self.model.policy.features_extractor(x)
#             pi_features, vf_features = self.model.policy.mlp_extractor(features)
#             action_net = self.model.policy.action_net(pi_features)
#             value_net = self.model.policy.value_net(vf_features)
            
#             return {
#                 'features_extractor': features,
#                 'pi_features': pi_features,
#                 'vf_features': vf_features,
#                 'action_net': action_net,
#                 'value_net': value_net
#             }
    
#     wrapped_model = LayerWrapper(model)

#     # Get layer output to determine number of neurons
#     with torch.no_grad():
#         test_output = wrapped_model(input_tensor)
#         layer_output = test_output[layer_name]
#         n_neurons = layer_output.shape[1]

#     # Initialize attributions array
#     attributions_all = np.zeros((n_neurons, len(input_tensor), input_tensor.shape[1]))
    
#     # Calculate gradients for each neuron without LayerIntegratedGradients
#     for i in range(n_neurons):
#         input_tensor_grad = input_tensor.clone().detach().requires_grad_()
        
#         output = wrapped_model(input_tensor_grad)[layer_name][:, i]
#         grad = torch.autograd.grad(output.sum(), input_tensor_grad)[0]
#         attributions = input_tensor_grad.detach() * grad.detach()
#         attributions_all[i] = attributions.detach().numpy()
    
#     avg_attributions = [np.mean(attributions_all[:,:,i]) for i in range(attributions_all.shape[2])]
    
#     # Plot the average attributions as a bar plot
#     plt.figure(figsize=(12, 6))
#     plt.bar(input_features, avg_attributions)
#     plt.xticks(range(len(input_features)), input_features, rotation=45, ha='right')
#     plt.xlabel('Features')
#     plt.ylabel('Average Attribution')
#     plt.title(f'Feature Importance in {layer_name} Layer')
#     plt.tight_layout()
#     plt.show()

#     return attributions_all, avg_attributions

# layer_names = [
#     'pi_features', 
#     'vf_features', 
#     'action_net', 
#     'value_net'
#     ]

# # Example usage
# neuron_activity, feature_attributions = analyze_layer_attributions(
#     model,
#     layer_names[0],  # 'pi_features'
#     torch.tensor(X_test, dtype=torch.float32)
# )