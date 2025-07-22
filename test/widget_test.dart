import 'package:flutter_test/flutter_test.dart';
import 'package:floorplan_detection_app/main.dart';

void main() {
  testWidgets('App launches without crashing', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const MainApp());

    // Verify that the app launches and shows the main screen
    expect(find.text('Floor Plan Room Detection'), findsOneWidget);
  });
}
