import VariationalAutoEncoder as VAE

trainer = VAE.Trainer()

vae = VAE.Model(
    VAE.Encoder(),
    VAE.Decoder(),
    epochs=25,
    optimizer='ADAM',
    learning_rate=0.01,
    reconstruction_alpha=0
)

vae.train(trainer.train_set, batch_size=32, preview=True)