<script lang="ts">
	import * as tf from "@tensorflow/tfjs";

	const modelPath = "model/ttt_model.json";
	let model: tf.LayersModel;

	const Players = {
		Empty: 0,
		Player1: 1,
		Player2: -1,
	} as const;
	type Player = typeof Players[keyof typeof Players];
	let user: Player;
	let bot: Player;
	let boxes = new Array<Player>(9).fill(Players.Empty);

	type Phase = "start" | "play" | "gameover";
	let phase: Phase = "start";

	let message = "";

	(async () => {
		await tf.ready();
		model = await tf.loadLayersModel(modelPath);
	})();

	function start(player: Player) {
		phase = "play";
		user = player;
		if (user === Players.Player1) {
			bot = Players.Player2;
		} else {
			bot = Players.Player1;
			predict();
		}
	}

	function update(index: number) {
		if (phase === "play" && boxes[index] === Players.Empty) {
			boxes[index] = user;
			result();
			predict();
		}
	}

	function predict() {
		if (phase === "play") {
			tf.tidy(() => {
				const matches = tf.stack([tf.tensor(boxes)]);
				const prediction = model.predict(matches) as tf.Tensor<tf.Rank>;
				const data = prediction.dataSync();
				while (data.some(Boolean)) {
					const index = data.indexOf(Math.max(...data));
					if (boxes[index] === Players.Empty) {
						boxes[index] = bot;
						result();
						return;
					}
					data[index] = null;
				}
			});
		}
	}

	function result() {
		[Players.Player1, Players.Player2].forEach((player) => {
			if (win(player)) {
				phase = "gameover";
				message = user === player ? "勝ち" : "負け";
			}
		});
	}

	function win(player: Player) {
		return (
			(boxes[0] === player && boxes[1] === player && boxes[2] === player) ||
			(boxes[3] === player && boxes[4] === player && boxes[5] === player) ||
			(boxes[6] === player && boxes[7] === player && boxes[8] === player) ||
			(boxes[0] === player && boxes[3] === player && boxes[6] === player) ||
			(boxes[1] === player && boxes[4] === player && boxes[7] === player) ||
			(boxes[2] === player && boxes[5] === player && boxes[8] === player) ||
			(boxes[0] === player && boxes[4] === player && boxes[8] === player) ||
			(boxes[2] === player && boxes[4] === player && boxes[6] === player)
		);
	}
</script>

<main>
	<h1>闇のゲーム</h1>

	{#if model}
		{#if phase === "start"}
			<div>
				<button on:click={() => start(Players.Player1)}>先攻</button>
				<button on:click={() => start(Players.Player2)}>後攻</button>
			</div>
		{:else}
			<div class="board">
				{#each boxes as box, index}
					<div on:click={() => update(index)} class="box">
						{#if box === Players.Player1}
							〇
						{:else if box === Players.Player2}
							×
						{/if}
					</div>
				{/each}
			</div>

			<p>{message}</p>
		{/if}
	{/if}
</main>

<style>
	.board {
		display: grid;
		grid-template-rows: 100px 100px 100px;
		grid-template-columns: 100px 100px 100px;
	}

	.box {
		border: 1px solid #555;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: xx-large;
	}
</style>
