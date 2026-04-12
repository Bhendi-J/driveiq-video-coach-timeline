import { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'

function Car({ features }) {
  const meshRef = useRef()
  const bodyRef = useRef()

  // Animate car tilt based on braking/lane change flags
  useFrame((state) => {
    if (!meshRef.current) return
    const t = state.clock.getElapsedTime()

    const brakingTilt  = (features?.braking_flag     || 0) * -0.08
    const laneTilt     = (features?.lane_change_flag  || 0) * 0.05
    const speedWobble  = Math.sin(t * 3) * 0.005 * Math.min((features?.speed || 0) / 100, 1)

    meshRef.current.rotation.x = brakingTilt + speedWobble
    meshRef.current.rotation.z = laneTilt
  })

  return (
    <group ref={meshRef}>
      {/* Car body */}
      <mesh ref={bodyRef} position={[0, 0, 0]}>
        <boxGeometry args={[2, 0.6, 1]} />
        <meshStandardMaterial color="#1e40af" metalness={0.8} roughness={0.2} />
      </mesh>

      {/* Roof */}
      <mesh position={[0, 0.55, 0]}>
        <boxGeometry args={[1.2, 0.4, 0.85]} />
        <meshStandardMaterial color="#1e3a8a" metalness={0.7} roughness={0.3} />
      </mesh>

      {/* Wheels */}
      {[[-0.65, -0.3, 0.5], [0.65, -0.3, 0.5], [-0.65, -0.3, -0.5], [0.65, -0.3, -0.5]].map(([x, y, z], i) => (
        <mesh key={i} position={[x, y, z]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.28, 0.28, 0.15, 16]} />
          <meshStandardMaterial color="#0f172a" metalness={0.3} roughness={0.8} />
        </mesh>
      ))}

      {/* Headlights */}
      {[[-0.3, 0, 0.52], [0.3, 0, 0.52]].map(([x, y, z], i) => (
        <mesh key={i} position={[x, y, z]}>
          <sphereGeometry args={[0.07, 8, 8]} />
          <meshStandardMaterial color="#fbbf24" emissive="#fbbf24" emissiveIntensity={2} />
        </mesh>
      ))}
    </group>
  )
}

export default function BehaviourVisualiser({ features }) {
  return (
    <div className="card">
      <div className="card-title">🚗 3D Behaviour Visualiser</div>
      <div className="three-wrap">
        <Canvas camera={{ position: [3, 2, 4], fov: 45 }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 8, 5]} intensity={1.2} />
          <pointLight position={[-3, 2, -3]} color="#3b82f6" intensity={0.6} />
          <Car features={features} />
          <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.8} />
          {/* Road surface */}
          <mesh position={[0, -0.6, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[20, 20]} />
            <meshStandardMaterial color="#111827" />
          </mesh>
        </Canvas>
      </div>
    </div>
  )
}
