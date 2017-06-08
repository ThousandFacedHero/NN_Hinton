function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> 
%   by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) 
%   matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. 
%   It's of the same shape as <rbm_w>.
    visible_data = sample_bernoulli(visible_data);
    
    Vis_Hid0 = visible_state_to_hidden_probabilities(rbm_w, visible_data);

    Vis_Hid0 = sample_bernoulli(Vis_Hid0);
    
    Hid_Vis1 = hidden_state_to_visible_probabilities(rbm_w, Vis_Hid0);

    Hid_Vis1 = sample_bernoulli(Hid_Vis1);

    % Calcul Vis -> Hid

    Vis_Hid1 = visible_state_to_hidden_probabilities(rbm_w, Hid_Vis1);

    Vis_Hid2 = sample_bernoulli(Vis_Hid1);

    % Calcul du gradient

    Gradient1 = configuration_goodness_gradient(visible_data, Vis_Hid0);

    Gradient2 = configuration_goodness_gradient(Hid_Vis1, Vis_Hid1);

    ret = Gradient1 - Gradient2;
    
    
end
