import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Loader } from 'lucide-react';
import { encodeWAV, downsampleBuffer } from '../utils/wavUtils';

interface AudioRecorderProps {
    onRecordingComplete: (audioBlob: Blob) => void;
    isProcessing: boolean;
    timeLimit?: number;
    onReRecord?: () => void;
    hasRecording?: boolean;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({
    onRecordingComplete,
    isProcessing,
    timeLimit,
    onReRecord,
    hasRecording = false,
}) => {
    const [isRecording, setIsRecording] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);
    const [timeRemaining, setTimeRemaining] = useState(timeLimit || 0);

    const streamRef = useRef<MediaStream | null>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<BlobPart[]>([]);
    // Ref mirrors isRecording so interval/onstop callbacks see the current value
    // without stale closure issues.
    const isRecordingRef = useRef(false);

    // Web Audio API refs — used only for waveform visualization
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const animationFrameRef = useRef<number | null>(null);

    const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

    useEffect(() => {
        return () => { cleanup(); };
    }, []);

    const cleanup = () => {
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }
        if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
        }
        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(t => t.stop());
            streamRef.current = null;
        }
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;

            // ── Waveform visualization via Web Audio API ─────────────────────
            const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
            const ctx = new AudioContextClass();
            await ctx.resume();
            audioContextRef.current = ctx;

            const source = ctx.createMediaStreamSource(stream);
            const analyser = ctx.createAnalyser();
            analyser.fftSize = 2048;
            analyserRef.current = analyser;

            // Connect analyser into the pull graph via a muted gain node so
            // getByteTimeDomainData actually receives data
            const muteGain = ctx.createGain();
            muteGain.gain.value = 0;
            source.connect(analyser);
            analyser.connect(muteGain);
            muteGain.connect(ctx.destination);

            // ── Audio capture via MediaRecorder (reliable cross-browser) ─────
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            chunksRef.current = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                const blob = new Blob(chunksRef.current, { type: mediaRecorder.mimeType });

                // Decode recorded audio → PCM using the browser's native decoder
                const arrayBuffer = await blob.arrayBuffer();
                const decodeCtx = new AudioContextClass();
                try {
                    const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
                    const pcm = audioBuffer.getChannelData(0); // mono
                    const resampled = downsampleBuffer(pcm, audioBuffer.sampleRate, 16000);
                    const wavBlob = encodeWAV(resampled, 16000);
                    onRecordingComplete(wavBlob);
                } catch (err) {
                    console.error('Audio processing failed:', err);
                    alert('Failed to process recording. Please click Re-record and try again.');
                } finally {
                    await decodeCtx.close();
                }
            };

            mediaRecorder.start(100); // collect chunks every 100ms

            isRecordingRef.current = true;
            setIsRecording(true);
            setRecordingTime(0);
            if (timeLimit) setTimeRemaining(timeLimit);

            setTimeout(() => drawWaveform(), 100);

            timerRef.current = setInterval(() => {
                setRecordingTime(prev => {
                    const next = prev + 1;
                    if (timeLimit && next >= timeLimit) stopRecording();
                    return next;
                });
                if (timeLimit) setTimeRemaining(prev => Math.max(0, prev - 1));
            }, 1000);

        } catch (err) {
            console.error('Microphone error:', err);
            alert('Please allow microphone access to record audio.');
        }
    };

    const stopRecording = () => {
        if (!isRecordingRef.current) return;
        isRecordingRef.current = false;
        setIsRecording(false);

        if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
        if (animationFrameRef.current) { cancelAnimationFrame(animationFrameRef.current); animationFrameRef.current = null; }

        // Stop MediaRecorder — triggers onstop which handles WAV encoding
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }

        // Stop visualization context and mic stream
        if (audioContextRef.current) { audioContextRef.current.close(); audioContextRef.current = null; }
        if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    };

    const drawWaveform = () => {
        if (!analyserRef.current || !canvasRef.current) return;

        const analyser = analyserRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const draw = () => {
            animationFrameRef.current = requestAnimationFrame(draw);
            analyser.getByteTimeDomainData(dataArray);

            const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
            gradient.addColorStop(0, 'rgb(243, 244, 246)');
            gradient.addColorStop(1, 'rgb(249, 250, 251)');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.lineWidth = 2.5;
            const waveGradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
            waveGradient.addColorStop(0, 'rgb(59, 130, 246)');
            waveGradient.addColorStop(0.5, 'rgb(37, 99, 235)');
            waveGradient.addColorStop(1, 'rgb(29, 78, 216)');
            ctx.strokeStyle = waveGradient;
            ctx.beginPath();

            const sliceWidth = canvas.width / bufferLength;
            let x = 0;
            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = (v * canvas.height) / 2;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                x += sliceWidth;
            }
            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();
        };

        draw();
    };

    const formatTime = (s: number) =>
        `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;

    return (
        <div className="flex flex-col items-center space-y-6">
            <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isProcessing}
                className={`
          w-32 h-32 rounded-full flex items-center justify-center
          transition-all duration-300 shadow-lg
          ${isRecording ? 'bg-red-500 hover:bg-red-600 animate-pulse' : 'bg-primary-600 hover:bg-primary-700'}
          ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
        `}
            >
                {isProcessing ? (
                    <Loader className="w-16 h-16 text-white animate-spin" />
                ) : isRecording ? (
                    <Square className="w-16 h-16 text-white" fill="white" />
                ) : (
                    <Mic className="w-16 h-16 text-white" />
                )}
            </button>

            {isRecording && (
                <div className="w-full max-w-md">
                    <canvas
                        ref={canvasRef}
                        width={600}
                        height={100}
                        className="w-full h-24 bg-gray-50 rounded-lg border-2 border-primary-200 shadow-inner"
                    />
                </div>
            )}

            {isRecording && (
                <div className="text-center">
                    {timeLimit ? (
                        <>
                            <div className="text-3xl font-bold text-gray-700">{formatTime(timeRemaining)}</div>
                            <p className="text-sm text-gray-500 mt-1">Time Remaining</p>
                        </>
                    ) : (
                        <div className="text-3xl font-bold text-gray-700">{formatTime(recordingTime)}</div>
                    )}
                </div>
            )}

            {hasRecording && !isRecording && !isProcessing && onReRecord && (
                <button
                    onClick={onReRecord}
                    className="px-6 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
                >
                    🔄 Re-record Answer
                </button>
            )}

            <div className="text-center max-w-md">
                {isProcessing ? (
                    <p className="text-gray-600">Processing your recording...</p>
                ) : isRecording ? (
                    <div className="space-y-2">
                        <p className="text-lg font-semibold text-red-600">🔴 Recording in progress</p>
                        <p className="text-sm text-gray-600">Click the button to stop</p>
                    </div>
                ) : (
                    <div className="space-y-2">
                        <p className="text-lg font-semibold text-gray-700">Ready to record</p>
                        <p className="text-sm text-gray-600">Click the microphone to start recording your response</p>
                        <p className="text-xs text-gray-500 mt-2">Speak for 30-90 seconds about a technical topic</p>
                    </div>
                )}
            </div>
        </div>
    );
};
