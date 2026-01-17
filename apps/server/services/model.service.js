export const getResult = (segment) => ({
  clipId: segment.clipId,
  timestamp: segment.timestamp,
  hitIndex: segment.hitIndex,
  videoPath: segment.videoPath,
  tracking: {
    person: {
      detected: true,
      pose_angles: {
        backswing: 87.5,
        hip_rotation: 45.2,
        shoulder_rotation: 92.3,
      },
    },
    club: {
      detected: true,
      speed_kmh: 145.3,
      impact_frame: 45,
    },
    ball: {
      detected: true,
      speed_kmh: 165.8,
      launch_angle: 14.5,
      trajectory: [
        { x: 320, y: 400 },
        { x: 350, y: 380 },
        { x: 400, y: 350 },
      ],
    },
  },
})