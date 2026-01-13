#!/usr/bin/env python3
"""Complex web task to demonstrate task decomposition."""

import asyncio
import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)

# A more realistic Express API with multiple files
WEB_APP_FILES = {
    "package.json": """{
  "name": "user-api",
  "version": "1.0.0",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "test": "jest"
  },
  "dependencies": {
    "express": "^4.18.2",
    "bcrypt": "^5.1.0"
  }
}""",

    "server.js": """const express = require('express');
const userRoutes = require('./routes/users');

const app = express();
app.use(express.json());

// Routes
app.use('/api/users', userRoutes);

// Error handler
app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ error: 'Internal server error' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

module.exports = app;
""",

    "routes/users.js": """const express = require('express');
const router = express.Router();
const UserService = require('../services/userService');

const userService = new UserService();

// GET all users
router.get('/', async (req, res) => {
  try {
    const users = await userService.getAllUsers();
    res.json(users);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// POST create user
// BUG: No input validation - accepts any data
router.post('/', async (req, res) => {
  try {
    const user = await userService.createUser(req.body);
    res.status(201).json(user);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// PUT update user
// BUG: No validation on email format
router.put('/:id', async (req, res) => {
  try {
    const user = await userService.updateUser(req.params.id, req.body);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    res.json(user);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// DELETE user
router.delete('/:id', async (req, res) => {
  try {
    const deleted = await userService.deleteUser(req.params.id);
    if (!deleted) {
      return res.status(404).json({ error: 'User not found' });
    }
    res.status(204).send();
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
""",

    "services/userService.js": """// Simple in-memory user service
class UserService {
  constructor() {
    this.users = [];
    this.nextId = 1;
  }

  async getAllUsers() {
    return this.users.map(u => ({ ...u, password: undefined }));
  }

  // BUG: Password stored in plain text, no hashing
  async createUser(data) {
    const user = {
      id: this.nextId++,
      email: data.email,
      name: data.name,
      password: data.password,  // Should be hashed!
      createdAt: new Date().toISOString()
    };
    this.users.push(user);
    return { ...user, password: undefined };
  }

  async updateUser(id, data) {
    const index = this.users.findIndex(u => u.id === parseInt(id));
    if (index === -1) return null;

    this.users[index] = { ...this.users[index], ...data };
    return { ...this.users[index], password: undefined };
  }

  async deleteUser(id) {
    const index = this.users.findIndex(u => u.id === parseInt(id));
    if (index === -1) return false;

    this.users.splice(index, 1);
    return true;
  }

  async findByEmail(email) {
    return this.users.find(u => u.email === email);
  }
}

module.exports = UserService;
""",

    "middleware/validate.js": """// Empty validation middleware - needs implementation
module.exports = {
  validateUser: (req, res, next) => {
    // TODO: Implement validation
    next();
  }
};
"""
}

# Complex task that should be decomposed into subtasks
COMPLEX_TASK = """
Implement comprehensive input validation and security improvements for the User API:

1. Add email validation for POST /api/users - email must be valid format (contains @ and domain)
2. Add email validation for PUT /api/users/:id - same validation as POST
3. Add name validation - name must be non-empty string, max 100 characters
4. Add password validation - minimum 8 characters for new users
5. Hash passwords using bcrypt before storing (the bcrypt library is already installed)

For all validation failures, return 400 status with descriptive error message in format:
{ "error": "Description of what's wrong" }
"""


async def main():
    print("=" * 70)
    print("  ATLAS Complex Web Task - Task Decomposition Demo")
    print("=" * 70)

    # Create temp repo with multiple files
    temp_dir = Path(tempfile.mkdtemp(prefix="atlas_complex_"))
    for name, content in WEB_APP_FILES.items():
        file_path = temp_dir / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    print(f"\nRepository: {temp_dir}")
    print(f"Files created:")
    for name in WEB_APP_FILES:
        print(f"  - {name}")

    # Load config
    from atlas.core.config import Config, set_config
    config = Config.from_env()
    set_config(config)
    print(f"\nModel: {config.model}")

    # Create orchestrator
    from atlas.core.dag_orchestrator import TaskDAGOrchestrator, TaskExecutionConfig, TaskDAGSubmission

    exec_config = TaskExecutionConfig(
        agents_per_task=2,
        top_k_per_task=1,
        beam_width=2,
        enable_quality_selection=False,
    )

    orchestrator = TaskDAGOrchestrator(
        config=config,
        execution_config=exec_config,
        use_agentic=False,
    )

    submission = TaskDAGSubmission(
        task_id="complex_web_task",
        description=COMPLEX_TASK,
        repo_path=temp_dir,
        max_tasks=5,  # Allow up to 5 subtasks
    )

    print("\n" + "=" * 70)
    print("  TASK DESCRIPTION")
    print("=" * 70)
    print(COMPLEX_TASK)

    print("\n" + "=" * 70)
    print("  RUNNING ATLAS")
    print("=" * 70)
    print(f"  Max Tasks: {submission.max_tasks}")
    print(f"  Agents per Task: {exec_config.agents_per_task}")
    print()

    result = await orchestrator.solve(submission)

    # Display task decomposition
    print("\n" + "=" * 70)
    print("  TASK DECOMPOSITION")
    print("=" * 70)

    if result.dag and result.dag.tasks:
        print(f"\nDecomposed into {len(result.dag.tasks)} tasks:\n")
        for i, (task_id, task_spec) in enumerate(result.dag.tasks.items(), 1):
            print(f"Task {i}: {task_spec.title}")
            print(f"  ID: {task_id}")
            print(f"  Description: {task_spec.description[:100]}...")
            print(f"  Contract: {task_spec.contract[:80]}...")
            print(f"  Files: {task_spec.ownership.allowed_files or task_spec.ownership.allowed_globs}")
            print(f"  Dependencies: {task_spec.dependencies or 'None'}")
            print()
    else:
        print("No task decomposition available")

    # Display results
    print("=" * 70)
    print("  EXECUTION RESULTS")
    print("=" * 70)
    print(f"\nStatus: {result.status}")
    print(f"Total Cost: ${result.cost_usd:.4f}")
    print(f"Total Time: {result.duration_seconds:.1f}s")

    if result.errors:
        print(f"\nErrors:")
        for err in result.errors:
            print(f"  - {err}")

    # Display final patch
    if result.final_patch:
        print("\n" + "=" * 70)
        print("  FINAL COMBINED PATCH")
        print("=" * 70)
        print(result.final_patch)
        print("=" * 70)
        print("\n[SUCCESS] Patch generated successfully!")
    else:
        print("\n[FAILED] No patch generated")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
