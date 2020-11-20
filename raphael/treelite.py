def export_model_treelite(model, extension='so', toolchain='gcc'):
    import treelite
    model.booster_.save_model('model.txt')
    treelite_model = treelite.Model.load('model.txt', model_format='lightgbm')
    treelite_model.export_lib(toolchain=toolchain, libpath=f'model.{extension}', params={
                              'parallel_comp': 32}, verbose=True)


def load_model_treelite(extension='so'):
    # TARGET MACHINE
    import treelite_runtime
    return treelite_runtime.Predictor(f'model.{extension}', verbose=True)


def predict_treelite(model, X):
    # TARGET MACHINE
    import treelite_runtime
    batch = treelite_runtime.Batch.from_npy2d(X.values)
    return predictor.predict(batch, pred_margin=False)
