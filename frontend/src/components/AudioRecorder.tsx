import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Loader } from 'lucide-react';
import { encodeWAV, downsampleBuffer } from '../utils/wavUtils';

interface AudioRecorderProps {
    onRecordingComplete: (audioBlob: Blob) => void;
    isProcessing: boolean;
    timeLimit?: number; // seconds, optional
    onReRecord?: () => void; // callback when re-record is requested
    hasRecording?: boolean; // whether a recording exists
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

    // Refs for AudioContext and processing
    const audioContextRef = useRef<AudioContext | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const processorRef = useRef<ScriptProcessorNode | null>(null);
    const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const chunksRef = useRef<Float32Array[]>([]);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const animationFrameRef = useRef<number | null>(null);

    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopRecordingContext();
        };
    }, []);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;

            // Create Audio Context
            const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
            const context = new AudioContext({ sampleRate: 44100 }); // Standard rate
            audioContextRef.current = context;

            const source = context.createMediaStreamSource(stream);
            sourceRef.current = source;

            // Capture mono channel (bufferSize 4096 is good balance)
            const processor = context.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;

            chunksRef.current = []; // Reset buffers

            processor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                // important: must copy the data, otherwise it gets garbage collected/overwritten
                chunksRef.current.push(new Float32Array(inputData));
            };

            // Create analyser for visualization
            const analyser = context.createAnalyser();
            analyser.fftSize = 2048;
            analyserRef.current = analyser;

            // Connect graph: source -> analyser -> processor -> destination
            source.connect(analyser);
            analyser.connect(processor);
            processor.connect(context.destination);

            setIsRecording(true);
            setRecordingTime(0);
            if (timeLimit) {
                setTimeRemaining(timeLimit);
            }

            // Start waveform visualization after a short delay to ensure canvas is rendered
            setTimeout(() => {
                drawWaveform();
            }, 100);

            // Start timer
            timerRef.current = setInterval(() => {
                setRecordingTime((prev) => {
                    const newTime = prev + 1;

                    // Auto-stop if time limit reached
                    if (timeLimit && newTime >= timeLimit) {
                        stopRecording();
                    }

                    return newTime;
                });

                // Update time remaining
                if (timeLimit) {
                    setTimeRemaining((prev) => Math.max(0, prev - 1));
                }
            }, 1000);

        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Please allow microphone access to record audio.');
        }
    };

    const stopRecordingContext = () => {
        // Stop animation
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }

        // Stop wrapper similar to stopRecording but just for cleanup
        if (processorRef.current) {
            processorRef.current.disconnect();
            processorRef.current = null;
        }
        if (analyserRef.current) {
            analyserRef.current.disconnect();
            analyserRef.current = null;
        }
        if (sourceRef.current) {
            sourceRef.current.disconnect();
            sourceRef.current = null;
        }
        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
    };

    const stopRecording = async () => {
        if (!isRecording) return;

        setIsRecording(false);
        if (timerRef.current) {
            clearInterval(timerRef.current);
        }

        // 1. Disconnect and stop
        stopRecordingContext();

        // 2. Process collected raw buffers
        if (chunksRef.current.length > 0) {
            // Flatten the array of Float32Arrays
            const totalLength = chunksRef.current.reduce((acc, curr) => acc + curr.length, 0);
            const mergedBuffer = new Float32Array(totalLength);
            let offset = 0;
            for (const chunk of chunksRef.current) {
                mergedBuffer.set(chunk, offset);
                offset += chunk.length;
            }

            // 3. Resample to 16kHz
            const targetRate = 16000;
            const resampledBuffer = downsampleBuffer(mergedBuffer, 44100, targetRate);

            // 4. Encode to WAV
            const wavBlob = encodeWAV(resampledBuffer, targetRate);
            onRecordingComplete(wavBlob);
        }
    };

    const drawWaveform = () => {
        if (!analyserRef.current || !canvasRef.current) {
            return;
        }

        const analyser = analyserRef.current;
        const canvas = canvasRef.current;
        const canvasCtx = canvas.getContext('2d');

        if (!canvasCtx) {
            return;
        }

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const draw = () => {
            // Request next frame first
            animationFrameRef.current = requestAnimationFrame(draw);

            // Get current audio data
            analyser.getByteTimeDomainData(dataArray);

            // Clear canvas with gradient background
            const gradient = canvasCtx.createLinearGradient(0, 0, 0, canvas.height);
            gradient.addColorStop(0, 'rgb(243, 244, 246)');
            gradient.addColorStop(1, 'rgb(249, 250, 251)');
            canvasCtx.fillStyle = gradient;
            canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw waveform
            canvasCtx.lineWidth = 2.5;
            const waveGradient = canvasCtx.createLinearGradient(0, 0, canvas.width, 0);
            waveGradient.addColorStop(0, 'rgb(59, 130, 246)'); // blue-500
            waveGradient.addColorStop(0.5, 'rgb(37, 99, 235)'); // primary-600
            waveGradient.addColorStop(1, 'rgb(29, 78, 216)'); // blue-700
            canvasCtx.strokeStyle = waveGradient;
            canvasCtx.beginPath();

            const sliceWidth = (canvas.width * 1.0) / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = (v * canvas.height) / 2;

                if (i === 0) {
                    canvasCtx.moveTo(x, y);
                } else {
                    canvasCtx.lineTo(x, y);
                }

                x += sliceWidth;
            }

            canvasCtx.lineTo(canvas.width, canvas.height / 2);
            canvasCtx.stroke();
        };

        draw();
    };

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div className="flex flex-col items-center space-y-6">
            {/* Recording Button */}
            <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isProcessing}
                className={`
          w-32 h-32 rounded-full flex items-center justify-center
          transition-all duration-300 shadow-lg
          ${isRecording
                        ? 'bg-red-500 hover:bg-red-600 animate-pulse'
                        : 'bg-primary-600 hover:bg-primary-700'
                    }
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

            {/* Waveform Visualization */}
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

            {/* Timer */}
            {isRecording && (
                <div className="text-center">
                    {timeLimit ? (
                        <>
                            <div className="text-3xl font-bold text-gray-700">
                                {formatTime(timeRemaining)}
                            </div>
                            <p className="text-sm text-gray-500 mt-1">Time Remaining</p>
                        </>
                    ) : (
                        <div className="text-3xl font-bold text-gray-700">
                            {formatTime(recordingTime)}
                        </div>
                    )}
                </div>
            )}

            {/* Re-record Button */}
            {hasRecording && !isRecording && !isProcessing && onReRecord && (
                <button
                    onClick={onReRecord}
                    className="px-6 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
                >
                    🔄 Re-record Answer
                </button>
            )}

            {/* Instructions */}
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
                        <p className="text-sm text-gray-600">
                            Click the microphone to start recording your response
                        </p>
                        <p className="text-xs text-gray-500 mt-2">
                            Speak for 30-90 seconds about a technical topic
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};

