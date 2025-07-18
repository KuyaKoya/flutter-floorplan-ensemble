import 'package:flutter/material.dart';
import 'screens/floorplan_detection_screen.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Floor Plan Room Detection',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const FloorPlanDetectionScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}
