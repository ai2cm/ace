import torch


class Validator:
    """
    Validator base class
    """

    def forward_grad(self, invar):
        pred_outvar = self.model(invar)
        return pred_outvar

    def forward_nograd(self, invar):
        with torch.no_grad():
            pred_outvar = self.model(invar)
        return pred_outvar

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        raise NotImplementedError("Subclass of Validator needs to implement this")

    @staticmethod
    def _l2_relative_error(true_var, pred_var):  # TODO replace with metric classes
        new_var = {}
        for key in true_var.keys():
            new_var["l2_relative_error_" + str(key)] = torch.sqrt(
                torch.mean(torch.square(true_var[key] - pred_var[key]))
                / torch.var(true_var[key])
            )
        return new_var
