
class Summary(object):
    def summary(self, model="sdid"):
        if model == "sdid":
            att = self.hat_tau(model="sdid")
            print("#------------------------------------------------------------")
            print("Syntetic Difference in Differences")
            print(f"point estimate: {att:.3f}")
            if self.sdid_se != None:
                print(
                    f"95% CI ({att - 1.96*self.sdid_se :.3f}, {att + 1.96*self.sdid_se:.3f})"
                )
