<script lang="ts">
	import { onMount, tick as svelteTick } from 'svelte';
	import { untrack } from 'svelte';
	import { WAVEFORM_HEIGHT } from '../lib/config.js';
	import { app } from '../lib/state.svelte.js';

	let {
		audio,
		playing = $bindable(false),
		time = $bindable(0),
		dur = $bindable(0)
	}: { audio: Blob; playing: boolean; time: number; dur: number } = $props();

	let canvas: HTMLCanvasElement;
	let peaks: number[] = [];
	let player: HTMLAudioElement | null = null;
	let url: string | null = null;
	let raf = 0;
	let dragging = false;
	let cw = 0;
	let ch = 0;

	// one-time setup: decode audio blob, compute peaks, create player element.
	onMount(() => {
		cw = canvas.clientWidth || 300;
		ch = WAVEFORM_HEIGHT;
		canvas.width = cw;
		canvas.height = ch;

		url = URL.createObjectURL(audio);
		player = new Audio(url);
		player.volume = untrack(() => app.volume);
		player.addEventListener('ended', () => {
			playing = false;
			time = 0;
			draw(0);
		});

		const actx = new AudioContext();
		audio
			.arrayBuffer()
			.then((buf) => actx.decodeAudioData(buf))
			.then((decoded) => {
				dur = decoded.duration;
				peaks = computePeaks(decoded, cw);
				draw(0);
				actx.close();
			})
			.catch(() => {});

		// touch events with passive:false so preventDefault stops Chromium from
		// scrolling (it silently ignores preventDefault on default-passive handlers).
		canvas.addEventListener('touchstart', preventTouch, { passive: false });
		canvas.addEventListener('touchmove', preventTouch, { passive: false });

		return () => {
			if (player) {
				player.pause();
				player = null;
			}
			if (url) {
				URL.revokeObjectURL(url);
				url = null;
			}
			cancelLoop();
			canvas.removeEventListener('touchstart', preventTouch);
			canvas.removeEventListener('touchmove', preventTouch);
		};
	});

	function preventTouch(e: TouchEvent) {
		e.preventDefault();
	}

	// redraw idle waveforms when the theme changes
	$effect(() => {
		app.dark; // track toggle
		svelteTick().then(() => {
			if (peaks.length > 0 && player) {
				draw(dur > 0 ? player.currentTime / dur : 0);
			}
		});
	});

	// play/pause toggle (driven by SongCard button via bind:playing)
	$effect(() => {
		const wantPlay = playing;
		if (!player) return;
		if (wantPlay) {
			player.volume = untrack(() => app.volume);
			player.play().catch(() => {});
			startLoop();
		} else {
			player.pause();
			cancelLoop();
		}
	});

	function computePeaks(buf: AudioBuffer, numBins: number): number[] {
		const raw = buf.getChannelData(0);
		const binSize = Math.floor(raw.length / numBins);
		const out: number[] = [];
		for (let i = 0; i < numBins; i++) {
			let max = 0;
			const start = i * binSize;
			const end = Math.min(start + binSize, raw.length);
			for (let j = start; j < end; j++) {
				const v = raw[j] < 0 ? -raw[j] : raw[j];
				if (v > max) max = v;
			}
			out.push(max);
		}
		return out;
	}

	// draw waveform with playback progress (0..1).
	// reads CSS colors fresh on every call (live dark/light theme support).
	function draw(progress: number) {
		if (!canvas || peaks.length === 0) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const style = getComputedStyle(canvas);
		const colorDim = style.getPropertyValue('--waveform-dim').trim() || '#555';
		const colorPlay = style.getPropertyValue('--waveform-play').trim() || '#2ed573';

		const mid = ch / 2;
		const barW = cw / peaks.length;

		ctx.clearRect(0, 0, cw, ch);

		for (let i = 0; i < peaks.length; i++) {
			const x = i * barW;
			const barH = peaks[i] * mid * 0.9;
			ctx.fillStyle = i / peaks.length <= progress ? colorPlay : colorDim;
			ctx.fillRect(x, mid - barH, Math.max(1, barW - 0.5), barH * 2);
		}

		if (progress > 0 && progress < 1) {
			ctx.fillStyle = colorPlay;
			ctx.fillRect(progress * cw - 0.5, 0, 1, ch);
		}
	}

	// rAF loop: sync volume, update time, redraw waveform
	function tick() {
		if (!player || player.paused) return;
		player.volume = app.volume;
		const t = player.currentTime;
		time = t;
		draw(dur > 0 ? t / dur : 0);
		raf = requestAnimationFrame(tick);
	}

	function startLoop() {
		cancelLoop();
		raf = requestAnimationFrame(tick);
	}

	function cancelLoop() {
		if (raf) {
			cancelAnimationFrame(raf);
			raf = 0;
		}
	}

	// seek to x position on the canvas
	function seekTo(clientX: number) {
		if (!player || !canvas || dur <= 0) return;
		const rect = canvas.getBoundingClientRect();
		const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
		player.currentTime = x * dur;
		time = player.currentTime;
		draw(x);
	}

	function onPointerDown(e: PointerEvent) {
		dragging = true;
		canvas.setPointerCapture(e.pointerId);
		seekTo(e.clientX);
	}

	function onPointerMove(e: PointerEvent) {
		if (!dragging) return;
		seekTo(e.clientX);
	}

	function onPointerUp() {
		dragging = false;
	}
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<canvas
	bind:this={canvas}
	class="waveform"
	onpointerdown={onPointerDown}
	onpointermove={onPointerMove}
	onpointerup={onPointerUp}
></canvas>

<style>
	.waveform {
		width: 100%;
		height: var(--waveform-h, 64px);
		cursor: pointer;
		border-radius: 2px;
		touch-action: none;
		user-select: none;
		-webkit-user-select: none;
	}
</style>
